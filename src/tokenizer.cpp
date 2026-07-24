// Aggregated multilingual tokenizer -- a C++ port of NeMo's
// AggregatedTTSTokenizer stack, replicating (verified against the NeMo
// checkout and golden encodings):
//
//   * tts_tokenizers.IPATokenizer.encode / encode_from_g2p
//       - en-US: english_text_preprocessing(lower=False) = NFD + strip
//         combining marks + synoglyph map (curly quotes -> ASCII);
//       - other locales: any_locale_text_preprocessing = NFC + U+2019 -> "'".
//       - encode_from_g2p appends EVERY symbol found in the vocab. Note that
//         spaces are NOT collapsed: ' ' is itself a vocab token, so the
//         "collapse" branch and the plain vocab branch both append it
//         (leading / duplicate spaces survive; only trailing spaces are
//         stripped). Unknown symbols are dropped with a warning.
//       - pad_with_space wraps the result in the space token.
//   * i18n_ipa.IpaG2p.__call__ + parse_one_word with phoneme_probability=1.0
//       - word regex (en: ASCII letters; any-locale: Latin+Indic+Korean),
//         '|...|' pass-through groups, punctuation chunks kept per-codepoint;
//       - set_grapheme_case (upper / lower / mixed) before every lookup;
//       - heteronyms -> graphemes; en-US "'s"/"s" suffix rules; dict lookup
//         (first pronunciation variant -- the GGUF blob stores exactly that);
//         mixed-case fallback to the uppercased word (de-DE);
//       - OOV -> graphemes with grapheme_prefix; unhandled hyphenated OOVs
//         are split on '-' and re-parsed per sub-word.
//   * tts_tokenizers.ArabicCharsTokenizer.encode (charset v1): per-codepoint
//     vocab lookup, spaces collapsed, trailing spaces stripped, padded.
//   * HF ByT5 (google/byt5-small) for fr/it/vi/ko: id = byte + 3, plus the
//     trailing </s> (local id 1) that HF's encode(add_special_tokens=True)
//     appends -- NeMo feeds the raw AutoTokenizer, so the </s> is part of the
//     reference token stream.
//
// zh (jieba) and ja (pyopenjtalk) need external segmenters and are reported
// as unsupported for now.
//
// Unicode support is intentionally minimal (no ICU): NFC/NFD are implemented
// with a hand-rolled table of Latin-1 Supplement / Latin Extended-A
// precomposed characters (enough for en/es/de/pt/fr/it), and uppercasing
// covers ASCII + Latin-1 + Latin Extended-A (with ss -> SS like Python's
// str.upper()). Hindi/Arabic/Vietnamese input is assumed to already be NFC;
// scripts outside these tables pass through unchanged.
#include "tokenizer.hpp"
#include "common.hpp"
#include "model_loader.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

// ---------------------------------------------------------------------------
// UTF-8 <-> codepoints
// ---------------------------------------------------------------------------

std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp  = 0;
        int      len = 1;
        if      (c < 0x80)            { cp = c; }
        else if ((c >> 5) == 0x06)    { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0x0E)    { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E)    { cp = c & 0x07; len = 4; }
        else { ++i; continue; }  // stray continuation byte: skip
        if (i + len > s.size()) { break; }
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            const unsigned char cc = (unsigned char)s[i + k];
            if ((cc >> 6) != 0x02) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { ++i; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

void utf8_append(std::string& s, uint32_t cp) {
    if      (cp < 0x80)    { s += (char)cp; }
    else if (cp < 0x800)   { s += (char)(0xC0 | (cp >> 6));
                             s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12));
                             s += (char)(0x80 | ((cp >> 6) & 0x3F));
                             s += (char)(0x80 | (cp & 0x3F)); }
    else                   { s += (char)(0xF0 | (cp >> 18));
                             s += (char)(0x80 | ((cp >> 12) & 0x3F));
                             s += (char)(0x80 | ((cp >> 6) & 0x3F));
                             s += (char)(0x80 | (cp & 0x3F)); }
}

std::string cp_str(uint32_t cp) {
    std::string s;
    utf8_append(s, cp);
    return s;
}

std::string encode_cps(const std::vector<uint32_t>& cps) {
    std::string s;
    for (uint32_t cp : cps) utf8_append(s, cp);
    return s;
}

// ---------------------------------------------------------------------------
// Minimal Unicode normalization (Latin-1 Supplement + Latin Extended-A)
// ---------------------------------------------------------------------------

struct latin_decomp { uint16_t pre, base, mark; };  // pre = base + combining mark

// Canonical (single-mark) decompositions. Characters without a canonical
// decomposition (AE, Eth, O-stroke, Thorn, sharp-s, D-stroke, dotless-i,
// kra, Eng, OE ligature, T-stroke, long-s, ...) are intentionally absent.
const latin_decomp k_latin_decomp[] = {
    {0x00C0,'A',0x300},{0x00C1,'A',0x301},{0x00C2,'A',0x302},{0x00C3,'A',0x303},
    {0x00C4,'A',0x308},{0x00C5,'A',0x30A},{0x00C7,'C',0x327},{0x00C8,'E',0x300},
    {0x00C9,'E',0x301},{0x00CA,'E',0x302},{0x00CB,'E',0x308},{0x00CC,'I',0x300},
    {0x00CD,'I',0x301},{0x00CE,'I',0x302},{0x00CF,'I',0x308},{0x00D1,'N',0x303},
    {0x00D2,'O',0x300},{0x00D3,'O',0x301},{0x00D4,'O',0x302},{0x00D5,'O',0x303},
    {0x00D6,'O',0x308},{0x00D9,'U',0x300},{0x00DA,'U',0x301},{0x00DB,'U',0x302},
    {0x00DC,'U',0x308},{0x00DD,'Y',0x301},
    {0x00E0,'a',0x300},{0x00E1,'a',0x301},{0x00E2,'a',0x302},{0x00E3,'a',0x303},
    {0x00E4,'a',0x308},{0x00E5,'a',0x30A},{0x00E7,'c',0x327},{0x00E8,'e',0x300},
    {0x00E9,'e',0x301},{0x00EA,'e',0x302},{0x00EB,'e',0x308},{0x00EC,'i',0x300},
    {0x00ED,'i',0x301},{0x00EE,'i',0x302},{0x00EF,'i',0x308},{0x00F1,'n',0x303},
    {0x00F2,'o',0x300},{0x00F3,'o',0x301},{0x00F4,'o',0x302},{0x00F5,'o',0x303},
    {0x00F6,'o',0x308},{0x00F9,'u',0x300},{0x00FA,'u',0x301},{0x00FB,'u',0x302},
    {0x00FC,'u',0x308},{0x00FD,'y',0x301},{0x00FF,'y',0x308},
    {0x0100,'A',0x304},{0x0101,'a',0x304},{0x0102,'A',0x306},{0x0103,'a',0x306},
    {0x0104,'A',0x328},{0x0105,'a',0x328},{0x0106,'C',0x301},{0x0107,'c',0x301},
    {0x0108,'C',0x302},{0x0109,'c',0x302},{0x010A,'C',0x307},{0x010B,'c',0x307},
    {0x010C,'C',0x30C},{0x010D,'c',0x30C},{0x010E,'D',0x30C},{0x010F,'d',0x30C},
    {0x0112,'E',0x304},{0x0113,'e',0x304},{0x0114,'E',0x306},{0x0115,'e',0x306},
    {0x0116,'E',0x307},{0x0117,'e',0x307},{0x0118,'E',0x328},{0x0119,'e',0x328},
    {0x011A,'E',0x30C},{0x011B,'e',0x30C},{0x011C,'G',0x302},{0x011D,'g',0x302},
    {0x011E,'G',0x306},{0x011F,'g',0x306},{0x0120,'G',0x307},{0x0121,'g',0x307},
    {0x0122,'G',0x327},{0x0123,'g',0x327},{0x0124,'H',0x302},{0x0125,'h',0x302},
    {0x0128,'I',0x303},{0x0129,'i',0x303},{0x012A,'I',0x304},{0x012B,'i',0x304},
    {0x012C,'I',0x306},{0x012D,'i',0x306},{0x012E,'I',0x328},{0x012F,'i',0x328},
    {0x0130,'I',0x307},
    {0x0134,'J',0x302},{0x0135,'j',0x302},{0x0136,'K',0x327},{0x0137,'k',0x327},
    {0x0139,'L',0x301},{0x013A,'l',0x301},{0x013B,'L',0x327},{0x013C,'l',0x327},
    {0x013D,'L',0x30C},{0x013E,'l',0x30C},
    {0x0143,'N',0x301},{0x0144,'n',0x301},{0x0145,'N',0x327},{0x0146,'n',0x327},
    {0x0147,'N',0x30C},{0x0148,'n',0x30C},
    {0x014C,'O',0x304},{0x014D,'o',0x304},{0x014E,'O',0x306},{0x014F,'o',0x306},
    {0x0150,'O',0x30B},{0x0151,'o',0x30B},
    {0x0154,'R',0x301},{0x0155,'r',0x301},{0x0156,'R',0x327},{0x0157,'r',0x327},
    {0x0158,'R',0x30C},{0x0159,'r',0x30C},{0x015A,'S',0x301},{0x015B,'s',0x301},
    {0x015C,'S',0x302},{0x015D,'s',0x302},{0x015E,'S',0x327},{0x015F,'s',0x327},
    {0x0160,'S',0x30C},{0x0161,'s',0x30C},{0x0162,'T',0x327},{0x0163,'t',0x327},
    {0x0164,'T',0x30C},{0x0165,'t',0x30C},
    {0x0168,'U',0x303},{0x0169,'u',0x303},{0x016A,'U',0x304},{0x016B,'u',0x304},
    {0x016C,'U',0x306},{0x016D,'u',0x306},{0x016E,'U',0x30A},{0x016F,'u',0x30A},
    {0x0170,'U',0x30B},{0x0171,'u',0x30B},{0x0172,'U',0x328},{0x0173,'u',0x328},
    {0x0174,'W',0x302},{0x0175,'w',0x302},{0x0176,'Y',0x302},{0x0177,'y',0x302},
    {0x0178,'Y',0x308},{0x0179,'Z',0x301},{0x017A,'z',0x301},{0x017B,'Z',0x307},
    {0x017C,'z',0x307},{0x017D,'Z',0x30C},{0x017E,'z',0x30C},
};

bool is_combining_mark(uint32_t cp) { return cp >= 0x300 && cp <= 0x36F; }

// precomposed -> base letter (used by the en-US NFD + strip-marks pass)
const std::unordered_map<uint32_t, uint32_t>& decomp_base_map() {
    static const std::unordered_map<uint32_t, uint32_t> m = [] {
        std::unordered_map<uint32_t, uint32_t> t;
        for (const auto& e : k_latin_decomp) t.emplace(e.pre, e.base);
        return t;
    }();
    return m;
}

// (base << 16 | mark) -> precomposed (used by the minimal NFC pass)
const std::unordered_map<uint32_t, uint32_t>& compose_map() {
    static const std::unordered_map<uint32_t, uint32_t> m = [] {
        std::unordered_map<uint32_t, uint32_t> t;
        for (const auto& e : k_latin_decomp)
            t.emplace(((uint32_t)e.base << 16) | e.mark, e.pre);
        return t;
    }();
    return m;
}

// english_text_preprocessing(text, lower=False): NFD + drop combining marks
// (Mn) + synoglyph -> ASCII. No lowercasing.
std::string english_pre(const std::string& text) {
    const auto& dec = decomp_base_map();
    std::string out;
    out.reserve(text.size());
    for (uint32_t cp : utf8_decode(text)) {
        auto it = dec.find(cp);
        if (it != dec.end()) cp = it->second;   // NFD, mark dropped
        if (is_combining_mark(cp)) continue;    // standalone Mn dropped
        if (cp == 0x2019) cp = '\'';            // right single quote
        else if (cp == 0x201C || cp == 0x201D) cp = '"';  // curly double quotes
        utf8_append(out, cp);
    }
    return out;
}

// any_locale_text_preprocessing: NFC (minimal Latin recomposition) + U+2019 -> '
std::string any_locale_pre(const std::string& text) {
    const auto& comp = compose_map();
    std::vector<uint32_t> cps = utf8_decode(text);
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    for (uint32_t cp : cps) {
        if (is_combining_mark(cp) && !out.empty()) {
            auto it = comp.find((out.back() << 16) | cp);
            if (it != comp.end()) { out.back() = it->second; continue; }
        }
        out.push_back(cp == 0x2019 ? (uint32_t)'\'' : cp);
    }
    return encode_cps(out);
}

// ---------------------------------------------------------------------------
// Python str.upper() over the supported subset (ASCII, Latin-1, Latin Ext-A)
// ---------------------------------------------------------------------------

std::string upper_str(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (uint32_t cp : utf8_decode(s)) {
        if (cp >= 'a' && cp <= 'z')                    { utf8_append(out, cp - 0x20); continue; }
        if (cp == 0x00DF)                              { out += "SS";                 continue; }  // sharp s
        if (cp >= 0x00E0 && cp <= 0x00FE && cp != 0x00F7) { utf8_append(out, cp - 0x20); continue; }
        if (cp == 0x00FF)                              { utf8_append(out, 0x0178);    continue; }
        if (cp == 0x0131)                              { utf8_append(out, 'I');       continue; }  // dotless i
        if (cp == 0x017F)                              { utf8_append(out, 'S');       continue; }  // long s
        if ((cp >= 0x0100 && cp <= 0x0137) || (cp >= 0x014A && cp <= 0x0177)) {
            utf8_append(out, (cp & 1) ? cp - 1 : cp);  continue;                                    // odd = lower
        }
        if (cp >= 0x0139 && cp <= 0x0148) { utf8_append(out, (cp & 1) ? cp : cp - 1); continue; }   // even = lower
        if (cp >= 0x0179 && cp <= 0x017E) { utf8_append(out, (cp & 1) ? cp : cp - 1); continue; }   // even = lower
        utf8_append(out, cp);  // Greek/Cyrillic/etc. pass through (not needed here)
    }
    return out;
}

std::string lower_ascii(const std::string& s) {
    std::string out = s;
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c += 0x20;
    return out;
}

// set_grapheme_case; note the only lowercasing NeMo applies at encode time is
// english_word_tokenize's ASCII word lowering, so "lower" stays ASCII-only.
std::string set_grapheme_case(const std::string& s, const std::string& gcase) {
    if (gcase == "upper") return upper_str(s);
    if (gcase == "lower") return lower_ascii(s);
    return s;  // "mixed"
}

// ---------------------------------------------------------------------------
// Character classes of the NeMo word regexes
// ---------------------------------------------------------------------------

bool is_ascii_letter(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}
bool is_latin_char(uint32_t cp) {  // LATIN_CHARS_ALL: A-Za-z + accented Latin-1
    return is_ascii_letter(cp) ||
           (cp >= 0x00C0 && cp <= 0x00D6) ||
           (cp >= 0x00D8 && cp <= 0x00F6) ||
           (cp >= 0x00F8 && cp <= 0x00FF);
}
bool is_indic_char(uint32_t cp) {  // INDIC_CHARS_ALL (danda excluded on purpose)
    return (cp >= 0x0900 && cp <= 0x0963) || (cp >= 0x0966 && cp <= 0x097F) ||
           (cp >= 0x0980 && cp <= 0x09FF) || (cp >= 0x0B80 && cp <= 0x0BFF) ||
           (cp >= 0x0C00 && cp <= 0x0C7F) || (cp >= 0x0C80 && cp <= 0x0CFF) ||
           (cp >= 0x0A80 && cp <= 0x0AFF);
}
bool is_korean_char(uint32_t cp) {  // KOREAN_CHARS
    return (cp >= 0xAC00 && cp <= 0xD7A3) || (cp >= 0x1100 && cp <= 0x11FF) ||
           (cp >= 0x3130 && cp <= 0x318F);
}
bool is_word_char_any(uint32_t cp) {  // WORD_CHARS_ALL
    return is_latin_char(cp) || is_indic_char(cp) || is_korean_char(cp);
}
// Python re \d (Unicode Nd) over the digit scripts these languages meet.
bool is_re_digit(uint32_t cp) {
    return (cp >= '0' && cp <= '9') ||
           (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F);
}

// ---------------------------------------------------------------------------
// Word tokenization (english_word_tokenize / any_locale_word_tokenize)
// ---------------------------------------------------------------------------

enum class chunk_kind { word, unchanged, punct };

struct text_chunk {
    chunk_kind kind;
    std::string text;                     // word or punct run
    std::vector<std::string> unchanged;   // '|...|' group split on ' '
};

// Replicates _WORDS_RE_EN / _WORDS_RE_ANY_LOCALE + _word_tokenize.
// A word is the longest run over {word_chars, '-', '\''} that starts at a
// word char; trailing '-'/'\'' are backed off. '|...|' is a pass-through
// group ('|' without a closing bar matches nothing and is dropped, like
// re.findall). Everything else accumulates into punct chunks.
std::vector<text_chunk> word_tokenize(const std::string& text, bool english) {
    auto is_word_char = [&](uint32_t cp) {
        return english ? is_ascii_letter(cp) : is_word_char_any(cp);
    };
    const std::vector<uint32_t> cps = utf8_decode(text);
    std::vector<text_chunk> out;
    size_t i = 0;
    while (i < cps.size()) {
        const uint32_t cp = cps[i];
        if (is_word_char(cp)) {
            size_t j = i;
            while (j < cps.size() &&
                   (is_word_char(cps[j]) || cps[j] == '-' || cps[j] == '\'')) ++j;
            while (j > i && !is_word_char(cps[j - 1])) --j;  // must end on a word char
            std::string w = encode_cps({cps.begin() + i, cps.begin() + j});
            if (english) w = lower_ascii(w);  // _word_tokenize(is_lower=True)
            out.push_back({chunk_kind::word, std::move(w), {}});
            i = j;
        } else if (cp == '|') {
            size_t j = i + 1;
            while (j < cps.size() && cps[j] != '|') ++j;
            if (j >= cps.size()) { ++i; continue; }  // no closing '|': dropped
            const std::string body = encode_cps({cps.begin() + i + 1, cps.begin() + j});
            text_chunk c{chunk_kind::unchanged, {}, {}};
            size_t pos = 0;  // Python str.split(" ")
            for (;;) {
                size_t sp = body.find(' ', pos);
                if (sp == std::string::npos) { c.unchanged.push_back(body.substr(pos)); break; }
                c.unchanged.push_back(body.substr(pos, sp - pos));
                pos = sp + 1;
            }
            out.push_back(std::move(c));
            i = j + 1;
        } else {
            size_t j = i;
            while (j < cps.size() && !is_word_char(cps[j]) && cps[j] != '|') ++j;
            out.push_back({chunk_kind::punct,
                           encode_cps({cps.begin() + i, cps.begin() + j}), {}});
            i = j;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// G2P dictionary blob ("word\tsym1 sym2 ...\n", lines sorted bytewise)
// ---------------------------------------------------------------------------

struct g2p_dict {
    std::string blob;
    std::vector<uint32_t> lines;  // offset of each line start

    void build(std::string data, const std::string& what) {
        blob = std::move(data);
        lines.clear();
        for (size_t pos = 0; pos < blob.size();) {
            lines.push_back((uint32_t)pos);
            size_t nl = blob.find('\n', pos);
            if (nl == std::string::npos)
                throw std::runtime_error(what + ": unterminated dict line");
            if (blob.find('\t', pos) > nl)
                throw std::runtime_error(what + ": dict line without a tab");
            pos = nl + 1;
        }
    }

    bool empty() const { return lines.empty(); }

    // strcmp(word-at-line, key) restricted to the "word" part before '\t'.
    int cmp_line(uint32_t off, const std::string& key) const {
        const char* p = blob.data() + off;
        size_t k = 0;
        while (*p != '\t') {
            if (k == key.size()) return 1;  // line word longer
            const unsigned char a = (unsigned char)*p, b = (unsigned char)key[k];
            if (a != b) return a < b ? -1 : 1;
            ++p; ++k;
        }
        return k == key.size() ? 0 : -1;  // line word is a prefix -> smaller
    }

    bool contains(const std::string& key) const { return find(key) >= 0; }

    int64_t find(const std::string& key) const {
        int64_t lo = 0, hi = (int64_t)lines.size() - 1;
        while (lo <= hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            const int c = cmp_line(lines[mid], key);
            if (c == 0)     return mid;
            else if (c < 0) lo = mid + 1;
            else            hi = mid - 1;
        }
        return -1;
    }

    // First (and only stored) pronunciation variant, split on ' '.
    bool lookup(const std::string& key, std::vector<std::string>& out) const {
        const int64_t line = find(key);
        if (line < 0) return false;
        out.clear();
        size_t pos = blob.find('\t', lines[line]) + 1;
        while (pos < blob.size() && blob[pos] != '\n') {
            size_t end = pos;
            while (end < blob.size() && blob[end] != ' ' && blob[end] != '\n') ++end;
            out.emplace_back(blob, pos, end - pos);
            pos = (end < blob.size() && blob[end] == ' ') ? end + 1 : end;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Sub-tokenizer state
// ---------------------------------------------------------------------------

struct sub_tokenizer {
    std::string name;
    std::string cls;                        // GGUF <name>.class
    std::string locale = "en-US";           // IPA only
    std::string grapheme_case = "upper";    // IpaG2p default
    std::string grapheme_prefix;            // "" or "#"
    int32_t offset = 0;
    int32_t size = 0;
    int32_t space_local = -1;
    bool pad_with_space = false;
    std::unordered_map<std::string, int32_t> vocab;  // token -> local id (last wins)
    g2p_dict dict;
    std::unordered_set<std::string> heteronyms;

    bool in_vocab(const std::string& tok) const { return vocab.count(tok) != 0; }
    int32_t id(const std::string& tok) const { return vocab.at(tok); }
};

std::vector<std::string> prefixed_graphemes(const sub_tokenizer& st, const std::string& word) {
    std::vector<std::string> out;
    for (uint32_t cp : utf8_decode(word)) out.push_back(st.grapheme_prefix + cp_str(cp));
    return out;
}

// IpaG2p.parse_one_word (phoneme_probability = 1.0: the keep-graphemes coin
// flip never fires). Returns handled=false only for the OOV grapheme
// fallback, which triggers the hyphen re-parse in the caller.
bool parse_one_word(const sub_tokenizer& st, const std::string& raw,
                    std::vector<std::string>& out) {
    const std::string word = set_grapheme_case(raw, st.grapheme_case);
    const std::vector<uint32_t> cps = utf8_decode(word);

    // Pure punctuation chunk: pass through per codepoint (no prefix).
    bool has_word_char = false;
    for (uint32_t cp : cps)
        if (is_word_char_any(cp) || is_re_digit(cp)) { has_word_char = true; break; }
    if (!has_word_char) {
        out.clear();
        for (uint32_t cp : cps) out.push_back(cp_str(cp));
        return true;
    }

    // Heteronyms stay graphemes.
    if (!st.heteronyms.empty() && st.heteronyms.count(word)) {
        out = prefixed_graphemes(st, word);
        return true;
    }

    // en-US possessive / plural suffix rules (ignore_ambiguous_words=false in
    // this checkpoint, so the is_unique guard is always satisfied).
    if (st.locale == "en-US" && !st.dict.empty()) {
        const size_t n = cps.size();
        auto ends_with = [&](const char* suf) {
            const size_t l = std::strlen(suf);
            return word.size() >= l && word.compare(word.size() - l, l, suf) == 0;
        };
        if (n > 2 && (ends_with("'s") || ends_with("'S")) && !st.dict.contains(word)) {
            const std::string base = word.substr(0, word.size() - 2);  // suffix is ASCII
            if (st.dict.lookup(base, out)) {
                const char last = base.empty() ? '\0' : base.back();
                if      (last == 'T' || last == 't') { out.push_back("s"); }
                else if (last == 'S' || last == 's') { out.push_back("ɪ"); out.push_back("z"); }
                else                                 { out.push_back("z"); }
                return true;
            }
        }
        if (n > 1 && (ends_with("s") || ends_with("S")) && !st.dict.contains(word)) {
            const std::string base = word.substr(0, word.size() - 1);
            if (st.dict.lookup(base, out)) {
                const char last = base.empty() ? '\0' : base.back();
                out.push_back((last == 'T' || last == 't') ? "s" : "z");
                return true;
            }
        }
    }

    // Dictionary lookup: first pronunciation variant (the blob stores it).
    if (!st.dict.empty() && st.dict.lookup(word, out)) return true;

    // grapheme_case == "mixed" (de-DE): retry with the uppercased word.
    if (st.grapheme_case == "mixed" && !st.dict.empty() &&
        st.dict.lookup(upper_str(word), out)) return true;

    // OOV: graphemes with prefix; unhandled (may be re-split on '-').
    out = prefixed_graphemes(st, word);
    return false;
}

// IpaG2p.__call__ minus the leading normalize_unicode_text (our preprocessing
// already leaves the text in the normal form this port supports).
std::vector<std::string> run_g2p(const sub_tokenizer& st, const std::string& text) {
    std::vector<std::string> prons;
    for (const text_chunk& ch : word_tokenize(text, st.locale == "en-US")) {
        if (ch.kind == chunk_kind::unchanged) {
            for (const std::string& w : ch.unchanged)
                prons.push_back(st.grapheme_prefix + w);
            continue;
        }
        std::vector<std::string> pron;
        const bool handled = parse_one_word(st, ch.text, pron);
        if (!handled && ch.text.find('-') != std::string::npos) {
            // Hyphenated OOV: split the ORIGINAL chunk on '-', re-parse parts.
            pron.clear();
            size_t pos = 0;
            for (;;) {
                const size_t dash = ch.text.find('-', pos);
                std::vector<std::string> sub;
                parse_one_word(st, ch.text.substr(pos, dash == std::string::npos
                                                       ? std::string::npos : dash - pos), sub);
                pron.insert(pron.end(), sub.begin(), sub.end());
                if (dash == std::string::npos) break;
                pron.push_back("-");
                pos = dash + 1;
            }
        }
        prons.insert(prons.end(), pron.begin(), pron.end());
    }
    return prons;
}

// IPATokenizer.encode_from_g2p. NOTE: spaces are NOT collapsed -- ' ' is a
// vocab token, so every occurrence is appended (verified against NeMo).
// Only trailing spaces are stripped, then pad_with_space wraps the result.
std::vector<int32_t> encode_from_g2p(const sub_tokenizer& st,
                                     const std::vector<std::string>& g2p_text) {
    std::vector<int32_t> ps;
    for (const std::string& p : g2p_text) {
        if (st.in_vocab(p)) {
            ps.push_back(st.id(p));
        } else if (p != " ") {
            MG_LOG("tokenizer %s: unknown symbol '%s' skipped", st.name.c_str(), p.c_str());
        }
    }
    while (!ps.empty() && ps.back() == st.space_local) ps.pop_back();
    if (st.pad_with_space) {
        ps.insert(ps.begin(), st.space_local);
        ps.push_back(st.space_local);
    }
    return ps;
}

std::vector<int32_t> encode_ipa(const sub_tokenizer& st, const std::string& text) {
    const std::string pre = st.locale == "en-US" ? english_pre(text) : any_locale_pre(text);
    return encode_from_g2p(st, run_g2p(st, pre));
}

// ArabicCharsTokenizer.encode: per-codepoint vocab lookup; unlike the IPA
// path this one DOES collapse repeated/leading spaces (c != space guard).
std::vector<int32_t> encode_arabic(const sub_tokenizer& st, const std::string& text) {
    std::vector<int32_t> cs;
    for (uint32_t cp : utf8_decode(any_locale_pre(text))) {
        const std::string c = cp_str(cp);
        if (c == " ") {
            if (!cs.empty() && cs.back() != st.space_local) cs.push_back(st.space_local);
        } else if (st.in_vocab(c)) {
            cs.push_back(st.id(c));
        } else {
            MG_LOG("tokenizer %s: unknown char '%s' skipped", st.name.c_str(), c.c_str());
        }
    }
    while (!cs.empty() && cs.back() == st.space_local) cs.pop_back();
    if (st.pad_with_space) {
        cs.insert(cs.begin(), st.space_local);
        cs.push_back(st.space_local);
    }
    return cs;
}

// HF ByT5 (google/byt5-small): <pad>=0, </s>=1, <unk>=2, byte b -> b + 3.
// encode(add_special_tokens=True) appends </s>; NeMo uses the tokenizer
// as-is, so the trailing </s> is part of the reference stream.
std::vector<int32_t> encode_byt5(const std::string& text) {
    std::vector<int32_t> ids;
    ids.reserve(text.size() + 1);
    for (unsigned char b : text) ids.push_back((int32_t)b + 3);
    ids.push_back(1);  // </s>
    return ids;
}

} // namespace

// ---------------------------------------------------------------------------
// magpie_tokenizer
// ---------------------------------------------------------------------------

struct magpie_tokenizer_state {
    std::vector<sub_tokenizer> subs;
    std::unordered_map<std::string, size_t> by_name;
    std::unordered_map<std::string, std::string> lang_to_name;
};

void magpie_tokenizer::init(const magpie_model& model) {
    if (!model.gguf) throw std::runtime_error("magpie_tokenizer::init: model not loaded");
    const magpie_tokenizer_hparams& th = model.hparams.tokenizer;
    auto st = std::make_shared<magpie_tokenizer_state>();

    mg::kv_reader kv(model.gguf);
    const std::vector<std::string> tokens = kv.require_arr_str("magpie.tokenizer.tokens");
    if (tokens.size() != th.vocab_size)
        kv.errors.push_back("magpie.tokenizer.tokens size != vocab_size");
    if (th.names.size() != th.offsets.size() || th.names.size() != th.sizes.size())
        kv.errors.push_back("tokenizer names/offsets/sizes length mismatch");
    kv.check("magpie_tokenizer::init");

    for (size_t i = 0; i < th.names.size(); ++i) {
        sub_tokenizer s;
        s.name   = th.names[i];
        s.offset = th.offsets[i];
        s.size   = th.sizes[i];
        const std::string p = "magpie.tokenizer." + s.name + ".";
        s.cls = kv.require_str((p + "class").c_str());
        if (kv.has((p + "pad_with_space").c_str()))
            s.pad_with_space = kv.require_bool((p + "pad_with_space").c_str());
        if (kv.has((p + "space_id_local").c_str()))
            s.space_local = kv.require_i32((p + "space_id_local").c_str());
        if (kv.has((p + "g2p.locale").c_str()))
            s.locale = kv.require_str((p + "g2p.locale").c_str());
        else if (kv.has((p + "locale").c_str()))
            s.locale = kv.require_str((p + "locale").c_str());
        if (kv.has((p + "g2p.grapheme_case").c_str()))
            s.grapheme_case = kv.require_str((p + "g2p.grapheme_case").c_str());
        if (kv.has((p + "g2p.grapheme_prefix").c_str()))
            s.grapheme_prefix = kv.require_str((p + "g2p.grapheme_prefix").c_str());

        if (s.offset < 0 || s.size < 0 || (size_t)(s.offset + s.size) > tokens.size())
            throw std::runtime_error("magpie_tokenizer::init: bad vocab slice for " + s.name);
        for (int32_t j = 0; j < s.size; ++j)
            s.vocab[tokens[s.offset + j]] = j;  // Python dict semantics: last wins

        const bool needs_space = s.cls == "IPATokenizer" || s.cls == "ArabicCharsTokenizer";
        if (needs_space &&
            (s.space_local < 0 || s.space_local >= s.size ||
             tokens[s.offset + s.space_local] != " "))
            throw std::runtime_error("magpie_tokenizer::init: bad space_id_local for " + s.name);

        st->by_name[s.name] = st->subs.size();
        st->subs.push_back(std::move(s));
    }
    kv.check("magpie_tokenizer::init");

    auto blob_of = [&](const std::string& tname) {
        ggml_tensor* t = model.require_tensor(tname);
        return std::string((const char*)t->data, (size_t)ggml_nbytes(t));
    };
    for (const std::string& n : th.g2p_names) {
        auto it = st->by_name.find(n);
        if (it == st->by_name.end()) continue;  // mandarin has a dict but no IPA path
        st->subs[it->second].dict.build(blob_of("g2p." + n + ".dict"), "g2p." + n);
    }
    for (const std::string& n : th.g2p_heteronym_names) {
        auto it = st->by_name.find(n);
        if (it == st->by_name.end()) continue;
        const std::string blob = blob_of("g2p." + n + ".heteronyms");
        auto& het = st->subs[it->second].heteronyms;
        for (size_t pos = 0; pos < blob.size();) {
            size_t nl = blob.find('\n', pos);
            if (nl == std::string::npos) nl = blob.size();
            if (nl > pos) het.emplace(blob, pos, nl - pos);
            pos = nl + 1;
        }
    }

    if (th.language_map_keys.size() != th.language_map_values.size())
        throw std::runtime_error("magpie_tokenizer::init: language map length mismatch");
    for (size_t i = 0; i < th.language_map_keys.size(); ++i)
        st->lang_to_name[th.language_map_keys[i]] = th.language_map_values[i];

    state = std::move(st);
}

std::vector<int32_t> magpie_tokenizer::encode(const std::string& text,
                                              const std::string& language) const {
    if (!state) throw std::runtime_error("magpie_tokenizer::encode: init() not called");
    const auto lang_it = state->lang_to_name.find(language);
    if (lang_it == state->lang_to_name.end())
        throw std::runtime_error("magpie_tokenizer: unknown language '" + language + "'");
    const auto sub_it = state->by_name.find(lang_it->second);
    if (sub_it == state->by_name.end())
        throw std::runtime_error("magpie_tokenizer: no tokenizer '" + lang_it->second + "'");
    const sub_tokenizer& st = state->subs[sub_it->second];

    std::vector<int32_t> local;
    if (st.cls == "IPATokenizer") {
        local = encode_ipa(st, text);
    } else if (st.cls == "ArabicCharsTokenizer") {
        local = encode_arabic(st, text);
    } else if (st.cls == "ByT5Tokenizer") {
        local = encode_byt5(text);
    } else if (st.cls == "ChinesePhonemesTokenizer" || st.cls == "JapanesePhonemeTokenizer") {
        throw std::runtime_error("magpie_tokenizer: language '" + language +
                                 "' (" + st.name + ") not supported yet: " + st.cls +
                                 " needs an external segmenter (jieba/pyopenjtalk)");
    } else {
        throw std::runtime_error("magpie_tokenizer: unsupported tokenizer class '" +
                                 st.cls + "' for language '" + language + "'");
    }

    for (int32_t& id : local) id += st.offset;
    return local;
}
