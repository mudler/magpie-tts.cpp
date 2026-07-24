// Tokenizer parity: exact integer-equality checks of magpie_tokenizer::encode
// against (a) golden encodings captured from the NeMo tokenizers (embedded
// below, no dump needed) and (b) the reference dump's tok.ids (text/language
// from the dump's ref.* KV, EOS appended like NeMo's chunk_text_for_inference).
//
// The golden vectors were produced by running the actual NeMo classes
// (IPATokenizer/IpaG2p with phoneme_probability=1.0, ArabicCharsTokenizer
// charset v1, HF google/byt5-small) on the extracted checkpoint resources and
// adding each sub-tokenizer's global offset. Note the ByT5 streams end in
// offset+1: HF encode(add_special_tokens=True) appends </s> (local id 1).
//
// Skips 77 when MAGPIE_MODEL is unset, and (after the self-contained golden
// checks pass) when MAGPIE_REF_DUMP is unset.
#include "parity.hpp"
#include "model_loader.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

static int g_fails = 0;

static std::string ids_str(const std::vector<int32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

static void check_ids(const magpie_tokenizer& tok, const char* lang, const char* text,
                      const std::vector<int32_t>& want) {
    std::vector<int32_t> got;
    try {
        got = tok.encode(text, lang);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] %s \"%s\": encode threw: %s\n", lang, text, e.what());
        ++g_fails;
        return;
    }
    if (got == want) {
        std::fprintf(stderr, "[ok]   %s \"%s\" (%zu ids)\n", lang, text, got.size());
    } else {
        std::fprintf(stderr, "[FAIL] %s \"%s\"\n  got  %s\n  want %s\n",
                     lang, text, ids_str(got).c_str(), ids_str(want).c_str());
        ++g_fails;
    }
}

static void check_throws(const magpie_tokenizer& tok, const char* lang, const char* text) {
    try {
        tok.encode(text, lang);
        std::fprintf(stderr, "[FAIL] %s: expected encode to throw\n", lang);
        ++g_fails;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ok]   %s throws: %s\n", lang, e.what());
    }
}

int main() {
    const std::string model_path = mgtest::env_or_skip("MAGPIE_MODEL");

    magpie_model model;
    magpie_tokenizer tok;
    try {
        model.load(model_path);
        tok.init(model);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] load/init: %s\n", e.what());
        return 1;
    }

    // --- self-contained golden checks (NeMo ground truth, no dump needed) ---

    // en (english_phoneme, offset 0, pad_with_space=false)
    // dict path + punctuation pass-through
    check_ids(tok, "en", "Hello world, this is a test of the text to speech system.",
        {55,79,90,59,62,87,93,90,68,82,59,52,5,93,90,75,84,64,93,90,84,69,93,90,
         88,93,90,65,81,64,65,93,90,88,67,93,75,79,93,90,65,81,58,64,65,93,90,65,
         66,93,90,64,63,56,65,86,93,90,64,84,64,65,79,60,7});
    // spaces are NOT collapsed (leading/duplicate kept, trailing stripped)
    check_ids(tok, "en", "a  b", {90,88,93,93,90,51,56});
    check_ids(tok, "en", " a",   {93,90,88});
    // en-US 's / s suffix rules + trailing-apostrophe punctuation
    check_ids(tok, "en", "Joe's airport's jones's airports cats' dogs",
        {90,52,89,62,87,69,93,90,81,85,91,63,78,85,65,64,93,90,52,89,62,87,61,69,
         84,69,93,90,81,85,91,63,78,85,65,64,93,90,58,74,65,64,2,93,90,52,77,83,69});
    // hyphenated dict word + hyphenated OOV split + plain OOV graphemes
    check_ids(tok, "en", "well-known xyzzyq-foo blarghle",
        {90,68,81,59,90,61,62,87,61,93,45,46,47,47,46,38,6,90,54,66,93,23,33,22,
         39,28,29,33,26});
    // digits ride inside the punctuation chunk, uppercase OOV -> graphemes
    check_ids(tok, "en", "NVIDIA GPU 123 test",
        {81,61,90,67,84,52,56,79,93,90,52,89,56,90,63,56,90,57,66,93,10,11,12,93,
         90,65,81,64,65});
    // heteronyms (read/bass/live) stay graphemes
    check_ids(tok, "en", "I read a book; the bass was live.",
        {90,50,84,93,39,26,22,25,93,90,88,93,90,51,87,58,20,93,75,79,93,23,22,40,
         40,93,90,68,77,69,93,33,30,43,26,7});
    // synoglyphs (curly quotes) + NFD-strip (cafe / naive)
    check_ids(tok, "en", "“quotes” and ’apostrophe’ café naïve",
        {1,90,58,68,62,87,65,64,1,93,79,61,52,93,2,79,90,63,77,64,65,85,79,91,54,
         56,2,93,58,79,90,54,53,84,93,91,61,50,84,90,56,67});
    // '|...|' pass-through group: unknown symbols dropped, space kept
    check_ids(tok, "en", "|EY1 EY1| unchanged",
        {93,79,61,90,65,86,53,84,61,52,89,52});
    check_ids(tok, "en", "'tis o'clock don't",
        {2,90,65,84,69,93,79,90,58,59,77,58,93,90,52,62,87,61,65});

    // es (spanish_phoneme, offset 480, pad_with_space=true): dict word
    // "hola" -> HOLA -> [ˈ,o,l,a], wrapped in spaces
    check_ids(tok, "es", "hola", {580,571,532,529,520,580});
    check_ids(tok, "es", "Hola mundo.",
        {580,571,532,529,520,580,530,571,537,531,522,532,487,580});
    check_ids(tok, "es", "niño pequeño",
        {580,531,571,526,564,532,580,533,523,528,571,523,564,532,580});
    // locale punctuation (inverted marks) + accented uppercase dict lookup
    check_ids(tok, "es", "¿Qué tal? ¡Bien!",
        {580,545,528,571,523,580,536,571,520,529,491,580,542,521,527,571,523,531,480,580});
    // hyphenated OOV re-split: both halves found in the dict
    check_ids(tok, "es", "España-Madrid xyzzyq-foo",
        {580,523,535,533,571,520,564,520,486,530,520,557,565,571,526,522,580,515,
         516,517,517,516,508,486,497,506,506,580});

    // de (german_phoneme, offset 583, grapheme_case=mixed, prefix '#')
    check_ids(tok, "de", "Hallo Welt.",
        {730,674,716,669,678,681,718,730,687,716,706,678,685,661,730});
    // mixed-case fallback: HALLO resolves via the uppercased dict entry
    check_ids(tok, "de", "HALLO", {730,674,716,669,678,681,718,730});
    check_ids(tok, "de", "Straße", {730,712,685,711,716,702,718,684,705,730});
    // OOV -> '#'-prefixed graphemes
    check_ids(tok, "de", "Xyzzyq foo", {730,608,635,636,636,635,627,730,616,625,625,730});
    check_ids(tok, "de", "Über ähnliche Bäume",
        {730,716,690,718,670,707,730,716,706,718,680,678,709,696,705,730,670,716,
         704,698,679,705,730});

    // pt-BR (offset 1017, grapheme_case=upper, prefix '#')
    check_ids(tok, "pt-BR", "Olá mundo.",
        {1125,1082,1079,1119,1070,1125,1080,1119,1099,1081,1072,1114,1063,1125});
    check_ids(tok, "pt-BR", "São Paulo é grande.",
        {1125,1085,1119,1100,1122,1114,1122,1125,1083,1119,1070,1114,1079,1114,
         1125,1119,1105,1125,1107,1111,1119,1100,1122,1081,1072,1117,1108,1063,1125});

    // hi (offset 1128): Devanagari dict + danda punctuation
    check_ids(tok, "hi", "नमस्ते दुनिया।",
        {1326,1190,1216,1189,1234,1195,1196,1182,1239,1326,1181,1232,1190,1223,
         1186,1178,1239,1322,1326});

    // ar-AE (offset 1329, char tokenizer, pad_with_space=true)
    check_ids(tok, "ar-AE", "مرحبا بالعالم",
        {1329,1405,1391,1387,1382,1381,1329,1382,1381,1404,1399,1381,1404,1405,1329});

    // byt5 languages: id = byte + 3 (+offset), trailing </s> = offset + 1
    // (HF ByT5 encode(add_special_tokens=True) appends </s>; NeMo keeps it).
    check_ids(tok, "fr", "abc", {1921,1922,1923,1822});
    check_ids(tok, "fr", "ça va être",
        {2019,1991,1921,1856,1942,1921,1856,2019,1994,1940,1938,1925,1822});
    check_ids(tok, "it", "Ciao mondo.",
        {2275,2313,2305,2319,2240,2317,2319,2318,2308,2319,2254,2206});
    check_ids(tok, "vi", "Xin chào thế giới",
        {2680,2697,2702,2624,2691,2696,2787,2752,2703,2624,2708,2696,2817,2778,
         2783,2624,2695,2697,2817,2779,2747,2697,2590});
    check_ids(tok, "ko", "안녕하세요",
        {3212,3125,3112,3211,3109,3125,3213,3125,3128,3212,3108,3160,3212,3130,
         3124,2974});

    // zh / ja / unknown languages must fail loudly
    check_throws(tok, "zh", "你好");
    check_throws(tok, "ja", "こんにちは");
    check_throws(tok, "xx", "hello");

    if (g_fails) {
        std::fprintf(stderr, "test_tokenizer_parity: %d golden check(s) FAILED\n", g_fails);
        return 1;
    }

    // --- reference-dump parity (text/language read from the dump itself) ---
    const std::string dump = mgtest::ref_dump_or_skip();

    std::string ref_text, ref_lang;
    if (!mgtest::load_kv_str(dump, "ref.text", ref_text) ||
        !mgtest::load_kv_str(dump, "ref.language", ref_lang)) return 1;

    std::vector<int32_t> ref_ids;
    if (!mgtest::load_baseline_i32(dump, "tok.ids", ref_ids)) return 1;

    std::vector<int32_t> got;
    try {
        got = tok.encode(ref_text, ref_lang);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] encode(ref.text): %s\n", e.what());
        return 1;
    }
    // NeMo appends EOS per chunk after encoding (chunk_text_for_inference).
    got.push_back((int32_t)model.hparams.text_eos_id);

    if (got != ref_ids) {
        std::fprintf(stderr,
            "[FAIL] tok.ids mismatch for %s \"%s\"\n  got  (%zu) %s\n  want (%zu) %s\n",
            ref_lang.c_str(), ref_text.c_str(),
            got.size(), ids_str(got).c_str(), ref_ids.size(), ids_str(ref_ids).c_str());
        return 1;
    }
    std::fprintf(stderr, "[ok]   tok.ids exact match (%zu ids incl. EOS %u) for %s \"%s\"\n",
                 got.size(), model.hparams.text_eos_id, ref_lang.c_str(), ref_text.c_str());
    std::fprintf(stderr, "test_tokenizer_parity: all checks passed\n");
    return 0;
}
