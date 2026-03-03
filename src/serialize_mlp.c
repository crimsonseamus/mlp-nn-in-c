#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "../include/serialize_mlp.h"

static int write_u32(FILE* f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1;
}

static int read_u32(FILE* f, uint32_t* out) {
    return fread(out, sizeof(uint32_t), 1, f) == 1;
}

int mlp_save(
    const char* path,
    const float* W1, const float* b1, int input_dim, int hidden_dim,
    const float* W2, const float* b2, int num_classes
) {
    if (!path || !W1 || !b1 || !W2 || !b2) return 0;
    if (input_dim <= 0 || hidden_dim <= 0 || num_classes <= 0) return 0;

    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    uint32_t magic = 0x4D4C5031u;   //"MLP1"
    uint32_t version = 1u;
    uint32_t in32 = (uint32_t)input_dim;
    uint32_t hid32 = (uint32_t)hidden_dim;
    uint32_t cls32 = (uint32_t)num_classes;

    if (!write_u32(f, magic) || !write_u32(f, version) || !write_u32(f, in32) || !write_u32(f, hid32) || !write_u32(f, cls32)) {
        fclose(f);
        return 0;
    }

    size_t w1n = (size_t)input_dim * (size_t)hidden_dim;
    size_t w2n = (size_t)hidden_dim * (size_t)num_classes;

    if (fwrite(W1, sizeof(float), w1n, f) != w1n) {
        fclose(f);
        return 0;
    }
    if (fwrite(b1, sizeof(float), (size_t)hidden_dim, f) != (size_t)hidden_dim) {
        fclose(f);
        return 0;
    }

    if (fwrite(W2, sizeof(float), w2n, f) != w2n) {
        fclose(f);
        return 0;

    }
    if (fwrite(b2, sizeof(float), (size_t)num_classes, f) != (size_t)num_classes) {
        fclose(f);
        return 0;

    }

    fclose(f);
    return 1;
}

int mlp_load(
    const char* path,
    float* W1, float* b1, int input_dim, int hidden_dim,
    float* W2, float* b2, int num_classes
) {
    if (!path || !W1 || !b1 || !W2 || !b2) return 0;
    if (input_dim <= 0 || hidden_dim <= 0 || num_classes <= 0) return 0;

    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t magic, version, in32, hid32, cls32;
    if (!read_u32(f, &magic) || !read_u32(f, &version) || !read_u32(f, &in32) || !read_u32(f, &hid32) || !read_u32(f, &cls32)) {
        fclose(f);
        return 0;
    }

    if (magic != 0x4D4C5031u) {
        fclose(f);
        return 0;
    }
    if (version != 1u) {
        fclose(f);
        return 0;
    }

    if (in32 != (uint32_t)input_dim || hid32 != (uint32_t)hidden_dim || cls32 != (uint32_t)num_classes) {
        fclose(f);
        return 0;
    }

    size_t w1n = (size_t)input_dim * (size_t)hidden_dim;
    size_t w2n = (size_t)hidden_dim * (size_t)num_classes;

    if (fread(W1, sizeof(float), w1n, f) != w1n) {
        fclose(f);
        return 0;
    }
    if (fread(b1, sizeof(float), (size_t)hidden_dim, f) != (size_t)hidden_dim) {
        fclose(f);
        return 0;

    }

    if (fread(W2, sizeof(float), w2n, f) != w2n) {
        fclose(f);
        return 0; }
    if (fread(b2, sizeof(float), (size_t)num_classes, f) != (size_t)num_classes) {
        fclose(f);
        return 0;

    }

    fclose(f);
    return 1;
}
