// ppl-dump: per-token log-probability dumper built on llama.cpp.
//
// Modes:
//   full     — tokenize a text file, run the model over it (up to --ctx
//              tokens), and emit one TSV row per token:
//                idx  token_id  n_bytes  logprob_nats
//              Token 0 has no prediction and gets logprob "nan". Byte counts
//              are UTF-8 piece lengths, so bits-per-byte curves can be
//              computed downstream without re-tokenizing.
//
//   variants — read a line-based job file with directives:
//                PREFIX <path>   extend the running context with this file
//                SPAN <id> <path> score this file as a continuation of the
//                                 current prefix, then roll the KV cache back
//              Each SPAN is scored with the same prefix (KV rollback via
//              llama_memory_seq_rm), and PREFIX chunks accumulate, so one
//              full-context decode serves many (depth, variant) probes.
//              Emits TSV rows:
//                span_id  tok_idx  token_id  n_bytes  logprob_nats
//              plus '# PREFIX n_tokens=<n> n_bytes=<b>' markers.
//
// Build (against a llama.cpp build tree):
//   g++ -O2 -std=c++17 ppl_dump.cpp -I<llama>/include -I<llama>/ggml/include \
//       -L<llama>/build/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,<llama>/build/bin -o ppl-dump

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path.c_str());
        exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool add_special) {
    // parse_special=false: source code containing special-token strings must
    // be treated as plain bytes, not control tokens.
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, false);
    std::vector<llama_token> toks(n);
    int m = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n, add_special, false);
    if (m < 0) {
        fprintf(stderr, "error: tokenization failed (%d)\n", m);
        exit(1);
    }
    toks.resize(m);
    return toks;
}

static int piece_bytes(const llama_vocab *vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    return n < 0 ? 0 : n;
}

// log softmax: returns logprob of `target` under `logits[0..n_vocab)`.
static double logprob_of(const float *logits, int n_vocab, llama_token target) {
    float maxl = logits[0];
    for (int i = 1; i < n_vocab; i++) maxl = logits[i] > maxl ? logits[i] : maxl;
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) sum += exp((double) logits[i] - maxl);
    return (double) logits[target] - maxl - log(sum);
}

struct Ctx {
    llama_model   *model  = nullptr;
    llama_context *lctx   = nullptr;
    const llama_vocab *vocab = nullptr;
    llama_batch    batch{};
    int n_vocab = 0;
    int n_batch = 0;
};

// Decode `toks[i0..i1)` at absolute positions starting at `pos0`, all with
// logits enabled. Calls `on_logits(local_idx, logits_row)` for every decoded
// token, in order. Returns 0 on success.
template <typename F>
static int decode_range(Ctx &C, const std::vector<llama_token> &toks, int i0, int i1, int pos0, F &&on_logits) {
    for (int s = i0; s < i1; s += C.n_batch) {
        int e = s + C.n_batch < i1 ? s + C.n_batch : i1;
        C.batch.n_tokens = e - s;
        for (int i = s; i < e; i++) {
            int k = i - s;
            C.batch.token[k]     = toks[i];
            C.batch.pos[k]       = pos0 + i - i0;
            C.batch.n_seq_id[k]  = 1;
            C.batch.seq_id[k][0] = 0;
            C.batch.logits[k]    = 1;
        }
        int rc = llama_decode(C.lctx, C.batch);
        if (rc != 0) {
            fprintf(stderr, "error: llama_decode rc=%d at token %d\n", rc, s);
            return rc;
        }
        for (int i = s; i < e; i++) {
            const float *logits = llama_get_logits_ith(C.lctx, i - s);
            on_logits(i, logits);
        }
    }
    return 0;
}

static void run_full(Ctx &C, const std::string &input, const std::string &output, int n_ctx) {
    std::string text = read_file(input);
    std::vector<llama_token> toks = tokenize(C.vocab, text, true);
    int n = (int) toks.size();
    if (n > n_ctx) { toks.resize(n_ctx); n = n_ctx; }
    fprintf(stderr, "full: %zu bytes -> %d tokens\n", text.size(), n);

    FILE *out = fopen(output.c_str(), "w");
    if (!out) { fprintf(stderr, "error: cannot write %s\n", output.c_str()); exit(1); }
    fprintf(out, "# mode=full n_tokens=%d input_bytes=%zu\n", n, text.size());
    fprintf(out, "idx\ttoken_id\tn_bytes\tlogprob_nats\n");

    // Row for token 0 (no prediction), then one row per predicted token.
    fprintf(out, "0\t%d\t%d\tnan\n", toks[0], piece_bytes(C.vocab, toks[0]));

    std::vector<double> pending_lp(n, 0.0);
    int rc = decode_range(C, toks, 0, n, 0, [&](int i, const float *logits) {
        if (i + 1 < n) {
            pending_lp[i + 1] = logprob_of(logits, C.n_vocab, toks[i + 1]);
        }
    });
    if (rc != 0) exit(2);
    for (int i = 1; i < n; i++) {
        fprintf(out, "%d\t%d\t%d\t%.6f\n", i, toks[i], piece_bytes(C.vocab, toks[i]), pending_lp[i]);
    }
    fclose(out);
}

static void run_variants(Ctx &C, const std::string &jobfile, const std::string &output, int n_ctx) {
    std::ifstream job(jobfile);
    if (!job) { fprintf(stderr, "error: cannot open %s\n", jobfile.c_str()); exit(1); }
    FILE *out = fopen(output.c_str(), "w");
    if (!out) { fprintf(stderr, "error: cannot write %s\n", output.c_str()); exit(1); }
    fprintf(out, "# mode=variants\n");
    fprintf(out, "span_id\ttok_idx\ttoken_id\tn_bytes\tlogprob_nats\n");

    llama_memory_t mem = llama_get_memory(C.lctx);

    int prefix_tokens = 0;
    long prefix_bytes = 0;
    bool first_chunk = true;
    // Logits of the last prefix token, needed to score the first span token
    // (a later llama_decode overwrites the context's logits buffer).
    std::vector<float> last_logits(C.n_vocab);
    bool have_last = false;

    std::string line;
    while (std::getline(job, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string cmd;
        ls >> cmd;
        if (cmd == "PREFIX") {
            std::string path;
            ls >> path;
            std::string text = read_file(path);
            std::vector<llama_token> toks = tokenize(C.vocab, text, first_chunk);
            first_chunk = false;
            if (prefix_tokens + (int) toks.size() > n_ctx) {
                fprintf(stderr, "error: prefix exceeds n_ctx (%d + %zu > %d)\n", prefix_tokens, toks.size(), n_ctx);
                exit(1);
            }
            int rc = decode_range(C, toks, 0, (int) toks.size(), prefix_tokens, [&](int i, const float *logits) {
                if (i == (int) toks.size() - 1) {
                    memcpy(last_logits.data(), logits, C.n_vocab * sizeof(float));
                    have_last = true;
                }
            });
            if (rc != 0) exit(2);
            prefix_tokens += (int) toks.size();
            prefix_bytes  += (long) text.size();
            fprintf(out, "# PREFIX n_tokens=%d n_bytes=%ld\n", prefix_tokens, prefix_bytes);
            fflush(out);
        } else if (cmd == "SPAN") {
            std::string id, path;
            ls >> id >> path;
            if (!have_last) { fprintf(stderr, "error: SPAN before any PREFIX\n"); exit(1); }
            std::string text = read_file(path);
            std::vector<llama_token> toks = tokenize(C.vocab, text, false);
            int ns = (int) toks.size();
            if (ns == 0) continue;
            if (prefix_tokens + ns > n_ctx) { fprintf(stderr, "error: span exceeds n_ctx\n"); exit(1); }

            std::vector<double> lp(ns, 0.0);
            lp[0] = logprob_of(last_logits.data(), C.n_vocab, toks[0]);
            if (ns > 1) {
                int rc = decode_range(C, toks, 0, ns - 1, prefix_tokens, [&](int i, const float *logits) {
                    lp[i + 1] = logprob_of(logits, C.n_vocab, toks[i + 1]);
                });
                if (rc != 0) exit(2);
                // Roll the KV cache back to the bare prefix.
                llama_memory_seq_rm(mem, 0, prefix_tokens, -1);
            }
            for (int i = 0; i < ns; i++) {
                fprintf(out, "%s\t%d\t%d\t%d\t%.6f\n", id.c_str(), i, toks[i], piece_bytes(C.vocab, toks[i]), lp[i]);
            }
            fflush(out);
        } else {
            fprintf(stderr, "error: unknown directive '%s'\n", cmd.c_str());
            exit(1);
        }
    }
    fclose(out);
}

int main(int argc, char **argv) {
    std::string model_path, mode, input, jobfile, output;
    int n_ctx = 16384, n_batch = 512, n_threads = 4;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "--model")   model_path = next();
        else if (a == "--mode")    mode = next();
        else if (a == "--input")   input = next();
        else if (a == "--job")     jobfile = next();
        else if (a == "--output")  output = next();
        else if (a == "--ctx")     n_ctx = atoi(next().c_str());
        else if (a == "--batch")   n_batch = atoi(next().c_str());
        else if (a == "--threads") n_threads = atoi(next().c_str());
        else { fprintf(stderr, "error: unknown arg %s\n", a.c_str()); exit(1); }
    }
    if (model_path.empty() || mode.empty() || output.empty()) {
        fprintf(stderr, "usage: ppl-dump --model M.gguf --mode full|variants (--input F | --job J) --output O [--ctx N] [--batch N] [--threads N]\n");
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_batch;
    cparams.n_ubatch        = n_batch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context *lctx = llama_init_from_model(model, cparams);
    if (!lctx) { fprintf(stderr, "error: failed to create context\n"); return 1; }

    Ctx C;
    C.model   = model;
    C.lctx    = lctx;
    C.vocab   = llama_model_get_vocab(model);
    C.n_vocab = llama_vocab_n_tokens(C.vocab);
    C.n_batch = n_batch;
    C.batch   = llama_batch_init(n_batch, 0, 1);

    if (mode == "full")          run_full(C, input, output, n_ctx);
    else if (mode == "variants") run_variants(C, jobfile, output, n_ctx);
    else { fprintf(stderr, "error: bad mode %s\n", mode.c_str()); return 1; }

    llama_batch_free(C.batch);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
