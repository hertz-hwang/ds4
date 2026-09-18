#include "ds4_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_layout(
        uint32_t height,
        uint32_t width,
        uint32_t start,
        const uint8_t *types,
        uint32_t type_count,
        const uint32_t *perm,
        uint32_t perm_count) {
    char error[160] = {0};
    ds4_deepseek4_image_layout layout = {0};
    if (!ds4_deepseek4_image_layout_build(
            &layout, height, width, start, error, sizeof(error))) {
        fprintf(stderr, "layout failed: %s\n", error);
        return 0;
    }
    int ok = layout.token_count == type_count &&
             layout.image_count == perm_count &&
             memcmp(layout.types, types, type_count) == 0 &&
             memcmp(layout.perm, perm,
                    (size_t)perm_count * sizeof(*perm)) == 0;
    if (!ok) fprintf(stderr, "layout differs for %ux%u at %u\n",
                     height, width, start);
    ds4_deepseek4_image_layout_free(&layout);
    return ok;
}

static int check_span_parser(void) {
    const int vocab = 100;
    const int tokens[] = {
        7,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_START,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE_NEWLINE,
        vocab + DS4_DEEPSEEK4_IMAGE_END,
        8,
        vocab + DS4_DEEPSEEK4_IMAGE_START,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_END,
        9,
    };
    uint32_t cursor = 0, block = 0, start = 0, end = 0;
    int found = ds4_deepseek4_next_image_span(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            &cursor, &block, &start, &end);
    if (found != 1 || block != 1 || start != 3 || end != 6 || cursor != 7)
        return 0;
    found = ds4_deepseek4_next_image_span(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            &cursor, &block, &start, &end);
    if (found != 1 || block != 8 || start != 8 || end != 10 || cursor != 11)
        return 0;
    if (ds4_deepseek4_next_image_span(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            &cursor, &block, &start, &end) != 0) return 0;

    static const int malformed[][4] = {
        {100 + DS4_DEEPSEEK4_IMAGE_PAD, 4, 5, 6},
        {100 + DS4_DEEPSEEK4_IMAGE_START, 4,
         100 + DS4_DEEPSEEK4_IMAGE_END, 6},
        {100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE_END, 6},
        {100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE, 6, 7},
        {100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE_END + 1, 6, 7},
    };
    for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); i++) {
        cursor = 0;
        if (ds4_deepseek4_next_image_span(
                malformed[i], 4, vocab, &cursor,
                &block, &start, &end) != -1) return 0;
    }

    uint32_t chunk = 0;
    if (!ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            0, 5, &chunk) || chunk != 1) return 0;
    if (!ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            1, 6, &chunk) || chunk != 6) return 0;
    if (ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            1, 5, &chunk)) return 0;
    if (ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            4, 8, &chunk)) return 0;
    return 1;
}

static int check_attention_bounds(void) {
    const int vocab = 100;
    const int tokens[] = {
        7,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_START,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE_END,
        8,
    };
    uint32_t bounds[sizeof(tokens) / sizeof(tokens[0]) * 2u];
    if (!ds4_deepseek4_attention_bounds(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            10u, 20u, 4u, bounds)) return 0;
    static const uint32_t expected[][2] = {
        {7, 10}, {8, 11}, {9, 18}, {10, 18}, {11, 18},
        {12, 18}, {12, 18}, {12, 18}, {12, 18}, {16, 19},
    };
    return memcmp(bounds, expected, sizeof(expected)) == 0;
}

/* The same 4x4 source, losslessly encoded as PNG and WebP: intake has to land
 * on one raster for both, so the decoded planes and their fingerprint agree
 * byte for byte.  Alpha is dropped the way PNG intake drops it, and an
 * animated container is refused before a frame is ever picked. */
static int check_webp_decode(void) {
static const uint8_t fixture_opaque_webp[] = {
    0x52, 0x49, 0x46, 0x46, 0x4a, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x4c,
    0x3e, 0x00, 0x00, 0x00, 0x2f, 0x03, 0xc0, 0x00, 0x00, 0x7f, 0x40, 0x90, 0x6d, 0x33, 0x88, 0xe3,
    0x0c, 0x7f, 0x68, 0x87, 0x10, 0x0b, 0xa6, 0x78, 0x77, 0x4c, 0x89, 0x05, 0x93, 0x33, 0x8b, 0xf9,
    0x73, 0x45, 0x32, 0xff, 0x6d, 0x58, 0x15, 0x42, 0x30, 0x59, 0x96, 0x31, 0x0f, 0x32, 0x10, 0x04,
    0x40, 0xc4, 0x39, 0x70, 0xe0, 0xc0, 0xa7, 0x03, 0x07, 0x0e, 0xa4, 0x44, 0xf4, 0x3f, 0x7c, 0x8d,
    0x45, 0x00,
};

static const size_t fixture_opaque_webp_len = 82;

static const uint8_t fixture_alpha_webp[] = {
    0x52, 0x49, 0x46, 0x46, 0x4c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x4c,
    0x40, 0x00, 0x00, 0x00, 0x2f, 0x03, 0xc0, 0x00, 0x10, 0x47, 0x40, 0x10, 0x40, 0x12, 0xc4, 0x38,
    0x83, 0x85, 0x30, 0xd8, 0x90, 0x64, 0xc2, 0x10, 0xbb, 0x65, 0x12, 0x4c, 0x40, 0xc1, 0xc8, 0xa4,
    0x6d, 0xf2, 0xcd, 0x54, 0xe5, 0x4f, 0xd8, 0xfc, 0x07, 0x48, 0x56, 0x79, 0x79, 0x51, 0x7a, 0x58,
    0xe6, 0xa0, 0x10, 0x92, 0x15, 0x3a, 0xf8, 0xbe, 0x3e, 0x21, 0x85, 0x10, 0x42, 0x11, 0xfd, 0x0f,
    0x66, 0x5e, 0xdd, 0x03,
};

static const size_t fixture_alpha_webp_len = 84;

static const uint8_t fixture_animated_webp[] = {
    0x52, 0x49, 0x46, 0x46, 0xe0, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x58,
    0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x41, 0x4e,
    0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46,
    0x56, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x64, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x3e, 0x00, 0x00, 0x00, 0x2f, 0x03, 0xc0, 0x00,
    0x00, 0x7f, 0x40, 0x90, 0x6d, 0x33, 0x88, 0xe3, 0x0c, 0x7f, 0x68, 0x87, 0x10, 0x0b, 0xa6, 0x78,
    0x77, 0x4c, 0x89, 0x05, 0x93, 0x33, 0x8b, 0xf9, 0x73, 0x45, 0x32, 0xff, 0x6d, 0x58, 0x15, 0x42,
    0x30, 0x59, 0x96, 0x31, 0x0f, 0x32, 0x10, 0x04, 0x40, 0xc4, 0x39, 0x70, 0xe0, 0xc0, 0xa7, 0x03,
    0x07, 0x0e, 0xa4, 0x44, 0xf4, 0x3f, 0x7c, 0x8d, 0x45, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x56, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x64, 0x00,
    0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x3e, 0x00, 0x00, 0x00, 0x2f, 0x03, 0xc0, 0x00, 0x00, 0x7f,
    0x40, 0x90, 0x6d, 0x33, 0x88, 0xe3, 0x0c, 0x7f, 0x68, 0x87, 0x10, 0x0b, 0xa6, 0x78, 0x77, 0x4c,
    0x89, 0x05, 0x93, 0x33, 0x8b, 0xf9, 0x73, 0x45, 0x32, 0xff, 0x6d, 0x58, 0x15, 0x42, 0x30, 0x59,
    0x96, 0x31, 0x0f, 0x32, 0x10, 0x04, 0x40, 0xc4, 0x39, 0x70, 0xe0, 0xc0, 0xa7, 0x03, 0x07, 0x0e,
    0xa4, 0x44, 0xf4, 0x3f, 0x7c, 0x8d, 0x45, 0x00,
};

static const size_t fixture_animated_webp_len = 232;

static const uint8_t fixture_opaque_png[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x08, 0x02, 0x00, 0x00, 0x00, 0x26, 0x93, 0x09,
    0x29, 0x00, 0x00, 0x00, 0x3d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xe0, 0x12, 0x61, 0x77,
    0x13, 0x51, 0x6d, 0x12, 0x71, 0xde, 0x27, 0x92, 0xc8, 0xc0, 0x15, 0xa0, 0xea, 0x16, 0xe0, 0xdc,
    0x14, 0x90, 0xb8, 0x2f, 0xa0, 0x9e, 0x81, 0xab, 0xc7, 0xd9, 0xad, 0x27, 0xb1, 0xa9, 0xa7, 0x7e,
    0x5f, 0xcf, 0x5c, 0x06, 0xae, 0x13, 0x89, 0x6e, 0x27, 0xea, 0x9b, 0x4e, 0xcc, 0xdd, 0x77, 0x62,
    0x37, 0x00, 0x82, 0xd3, 0x13, 0x31, 0xc9, 0xd0, 0x57, 0xf4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

static const size_t fixture_opaque_png_len = 118;


    char error[160] = {0};
    ds4_image png_raster = {0};
    ds4_image webp_raster = {0};
    ds4_image alpha_raster = {0};
    ds4_image animated = {0};

    if (!ds4_image_decode_memory(&png_raster, fixture_opaque_png,
                                 fixture_opaque_png_len, error, sizeof(error))) {
        fprintf(stderr, "PNG fixture did not decode: %s\n", error);
        return 0;
    }
#ifdef DS4_HAVE_WEBP
    if (!ds4_image_decode_memory(&webp_raster, fixture_opaque_webp,
                                 fixture_opaque_webp_len, error, sizeof(error))) {
        fprintf(stderr, "lossless WebP fixture did not decode: %s\n", error);
        ds4_image_free(&png_raster);
        return 0;
    }
    int ok = webp_raster.width == 4u && webp_raster.height == 4u &&
             memcmp(webp_raster.rgb, png_raster.rgb, 4u * 4u * 3u) == 0 &&
             memcmp(webp_raster.fingerprint, png_raster.fingerprint,
                    sizeof(webp_raster.fingerprint)) == 0;
    if (!ok) fprintf(stderr, "WebP raster differs from the PNG of the same source\n");

    if (ok && !ds4_image_decode_memory(&alpha_raster, fixture_alpha_webp,
                                       fixture_alpha_webp_len, error,
                                       sizeof(error))) {
        fprintf(stderr, "WebP with an alpha plane did not decode: %s\n", error);
        ok = 0;
    } else if (ok && (alpha_raster.width != 4u || alpha_raster.height != 4u)) {
        fprintf(stderr, "WebP alpha fixture decoded to %ux%u\n",
                alpha_raster.width, alpha_raster.height);
        ok = 0;
    }

    error[0] = '\0';
    if (ok && ds4_image_decode_memory(&animated, fixture_animated_webp,
                                      fixture_animated_webp_len, error,
                                      sizeof(error))) {
        fprintf(stderr, "animated WebP was accepted\n");
        ok = 0;
    } else if (ok && !strstr(error, "animated")) {
        fprintf(stderr, "animated WebP refusal did not name the cause: %s\n", error);
        ok = 0;
    }
#else
    /* Without libwebp every WebP container is refused, and the refusal names
     * the missing dependency instead of blaming the payload's shape. */
    int ok = 1;
    const struct {
        const uint8_t *bytes;
        size_t len;
    } refused[] = {
        { fixture_opaque_webp, fixture_opaque_webp_len },
        { fixture_alpha_webp, fixture_alpha_webp_len },
        { fixture_animated_webp, fixture_animated_webp_len },
    };
    for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); i++) {
        error[0] = '\0';
        if (ds4_image_decode_memory(&webp_raster, refused[i].bytes,
                                    refused[i].len, error, sizeof(error)) ||
            !strstr(error, "libwebp")) {
            fprintf(stderr,
                    "WebP refusal without libwebp is unclear: %s\n", error);
            ok = 0;
        }
    }
#endif
    ds4_image_free(&png_raster);
    ds4_image_free(&webp_raster);
    ds4_image_free(&alpha_raster);
    ds4_image_free(&animated);
    return ok;
}

int main(void) {
    static const uint8_t types_a[] = {
        1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 4,
    };
    static const uint32_t perm_a[] = {0, 3, 1, 4, 2, 5};
    static const uint8_t types_b[] = {
        1, 1, 0, 2, 2, 2, 2, 3, 3, 2, 1, 2, 1, 3, 1, 4,
    };
    static const uint32_t perm_b[] = {0, 2, 1, 3, 4, 5};
    static const uint8_t types_c[] = {0, 2, 1, 3, 1, 4};
    static const uint32_t perm_c[] = {0};
    if (!check_layout(2, 3, 0, types_a, sizeof(types_a),
                      perm_a, sizeof(perm_a) / sizeof(perm_a[0])) ||
        !check_layout(3, 2, 5, types_b, sizeof(types_b),
                      perm_b, sizeof(perm_b) / sizeof(perm_b[0])) ||
        !check_layout(1, 1, 3, types_c, sizeof(types_c),
                      perm_c, sizeof(perm_c) / sizeof(perm_c[0])) ||
        !check_span_parser() ||
        !check_attention_bounds() ||
        !check_webp_decode()) {
        return 1;
    }

    ds4_image image = {
        .width = 17,
        .height = 9,
    };
    image.rgb = malloc((size_t)image.width * image.height * 3u);
    if (!image.rgb) return 1;
    for (uint32_t y = 0; y < image.height; y++) {
        for (uint32_t x = 0; x < image.width; x++) {
            uint8_t *pixel = image.rgb + ((size_t)y * image.width + x) * 3u;
            pixel[0] = (uint8_t)(x * 13u + y * 3u);
            pixel[1] = (uint8_t)(x * 5u + y * 17u);
            pixel[2] = (uint8_t)(x * 7u + y * 11u);
        }
    }
    char error[160] = {0};
    ds4_deepseek4_image_patches patches = {0};
    if (!ds4_image_preprocess_deepseek4(
            &patches, &image, error, sizeof(error))) {
        fprintf(stderr, "preprocess failed: %s\n", error);
        free(image.rgb);
        return 1;
    }
    int ok = patches.padded_width == 532u &&
             patches.padded_height == 280u &&
             patches.content_width == 529u &&
             patches.content_height == 280u &&
             patches.grid_width == 38u &&
             patches.grid_height == 20u &&
             patches.llm_grid_width == 13u &&
             patches.llm_grid_height == 7u &&
             patches.patch_count == 760u;
    const size_t values = (size_t)patches.patch_count * 588u;
    for (size_t i = 0; ok && i < values; i++) {
        ok = isfinite(patches.patches[i]) &&
             patches.patches[i] >= -1.00001f &&
             patches.patches[i] <= 1.00001f;
    }
    if (!ok) fprintf(stderr, "DeepSeek preprocessing dimensions or values differ\n");
    ds4_deepseek4_image_patches_free(&patches);
    free(image.rgb);
    return ok ? 0 : 1;
}
