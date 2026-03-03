#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/gui.h"
#include "../include/serialize_mlp.h"
#include "../include/model_mlp.h"
#include "../include/math_ops.h"

static void canvas_clear(uint8_t *canvas, int w, int h) { memset(canvas, 0, (size_t)w * (size_t)h); }

static inline uint8_t u8_clamp_int(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static void draw_filled_rect(SDL_Renderer *r, int x, int y, int w, int h) {
    SDL_Rect rc = { x, y, w, h };
    SDL_RenderFillRect(r, &rc);
}

static const uint8_t* glyph5x7(char ch) {
    static const uint8_t sp[7] = {0,0,0,0,0,0,0};

    static const uint8_t g0[7] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
    static const uint8_t g1[7] = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E};
    static const uint8_t g2[7] = {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F};
    static const uint8_t g3[7] = {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E};
    static const uint8_t g4[7] = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02};
    static const uint8_t g5[7] = {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E};
    static const uint8_t g6[7] = {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E};
    static const uint8_t g7[7] = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08};
    static const uint8_t g8[7] = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E};
    static const uint8_t g9[7] = {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C};

    static const uint8_t gd[7] = {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C};
    static const uint8_t gc[7] = {0x00,0x04,0x04,0x00,0x04,0x04,0x00};
    static const uint8_t gp[7] = {0x19,0x1A,0x04,0x08,0x16,0x06,0x00};

    switch (ch) {
        case '0': return g0; case '1': return g1; case '2': return g2; case '3': return g3; case '4': return g4;
        case '5': return g5; case '6': return g6; case '7': return g7; case '8': return g8; case '9': return g9;
        case '.': return gd;
        case ':': return gc;
        case '%': return gp;
        case ' ': return sp;
        default:  return sp;
    }
}

static void draw_text5x7(SDL_Renderer *r, int x, int y, int scale, const char *s) {
    int cx = x;
    for (const char *p = s; *p; p++) {
        const uint8_t *g = glyph5x7(*p);
        for (int row = 0; row < 7; row++) {
            uint8_t bits = g[row];
            for (int col = 0; col < 5; col++) {
                if (bits & (1u << (4 - col))) {
                    draw_filled_rect(r, cx + col * scale, y + row * scale, scale, scale);
                }
            }
        }
        cx += 6 * scale;
    }
}

static void canvas_stamp_brush(uint8_t *canvas, int w, int h, int cx, int cy, int r, int strength, int erase) {
    if (r < 1) r = 1;
    if (strength < 1) strength = 1;
    if (strength > 255) strength = 255;

    int y0 = cy - r, y1 = cy + r;
    int x0 = cx - r, x1 = cx + r;
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    if (y1 >= h) y1 = h - 1;
    if (x1 >= w) x1 = w - 1;

    float rf = (float)r;
    float inv = 1.0f / (rf + 1e-6f);

    for (int y = y0; y <= y1; y++) {
        float dy = (float)(y - cy);
        for (int x = x0; x <= x1; x++) {
            float dx = (float)(x - cx);
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist > rf) continue;

            float t = 1.0f - dist * inv;
            float wgt = t * t;
            int delta = (int)lroundf(wgt * (float)strength);

            int idx = y * w + x;
            int cur = (int)canvas[idx];
            int next = erase ? (cur - delta) : (cur + delta);
            canvas[idx] = u8_clamp_int(next);
        }
    }
}

static void canvas_to_rgba(const uint8_t *canvas, int w, int h, uint32_t *out_rgba) {
    for (int i = 0; i < w * h; i++) {
        uint8_t v = canvas[i];
        out_rgba[i] = (0xFFu << 24) | ((uint32_t)v << 16) | ((uint32_t)v << 8) | (uint32_t)v;
    }
}

static int find_bbox(const uint8_t *canvas, int w, int h, uint8_t thresh,
                     int *x_min, int *y_min, int *x_max, int *y_max) {
    int xmin = w, ymin = h, xmax = -1, ymax = -1;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t v = canvas[y * w + x];
            if (v > thresh) {
                if (x < xmin) xmin = x;
                if (x > xmax) xmax = x;
                if (y < ymin) ymin = y;
                if (y > ymax) ymax = y;
            }
        }
    }
    if (xmax < 0) return 0;
    *x_min = xmin; *y_min = ymin; *x_max = xmax; *y_max = ymax;
    return 1;
                     }

                     static float sample_bilinear_u8(const uint8_t *img, int w, int h, float x, float y) {
                         if (x < 0.0f) x = 0.0f;
                         if (y < 0.0f) y = 0.0f;
                         if (x > (float)(w - 1)) x = (float)(w - 1);
                         if (y > (float)(h - 1)) y = (float)(h - 1);

                         int x0 = (int)floorf(x);
                         int y0 = (int)floorf(y);
                         int x1 = x0 + 1; if (x1 >= w) x1 = w - 1;
                         int y1 = y0 + 1; if (y1 >= h) y1 = h - 1;

                         float fx = x - (float)x0;
                         float fy = y - (float)y0;

                         float v00 = (float)img[y0 * w + x0];
                         float v10 = (float)img[y0 * w + x1];
                         float v01 = (float)img[y1 * w + x0];
                         float v11 = (float)img[y1 * w + x1];

                         float v0 = v00 + fx * (v10 - v00);
                         float v1 = v01 + fx * (v11 - v01);
                         return v0 + fy * (v1 - v0);
                     }

                     static void blur28_inplace(float *img28) {
                         float tmp[28 * 28];
                         for (int i = 0; i < 28 * 28; i++) tmp[i] = img28[i];

                         for (int y = 0; y < 28; y++) {
                             for (int x = 0; x < 28; x++) {
                                 float sum = 0.0f, wsum = 0.0f;
                                 for (int dy = -1; dy <= 1; dy++) {
                                     int yy = y + dy; if (yy < 0) yy = 0; if (yy > 27) yy = 27;
                                     for (int dx = -1; dx <= 1; dx++) {
                                         int xx = x + dx; if (xx < 0) xx = 0; if (xx > 27) xx = 27;
                                         float w = (dx == 0 && dy == 0) ? 4.0f : ((dx == 0 || dy == 0) ? 2.0f : 1.0f);
                                         sum += w * tmp[yy * 28 + xx];
                                         wsum += w;
                                     }
                                 }
                                 img28[y * 28 + x] = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
                             }
                         }
                     }

                     static void shift28(const float *src, float *dst, int sx, int sy) {
                         for (int i = 0; i < 28 * 28; i++) dst[i] = 0.0f;
                         for (int y = 0; y < 28; y++) {
                             for (int x = 0; x < 28; x++) {
                                 int nx = x + sx;
                                 int ny = y + sy;
                                 if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) dst[ny * 28 + nx] = src[y * 28 + x];
                             }
                         }
                     }

                     static void center_of_mass_shift28(float *img28) {
                         double sum = 0.0, sx = 0.0, sy = 0.0;
                         for (int y = 0; y < 28; y++) {
                             for (int x = 0; x < 28; x++) {
                                 double v = (double)img28[y * 28 + x];
                                 sum += v;
                                 sx += v * (double)x;
                                 sy += v * (double)y;
                             }
                         }
                         if (sum <= 1e-12) return;

                         double cx = sx / sum;
                         double cy = sy / sum;

                         int dx = (int)lround(14.0 - cx);
                         int dy = (int)lround(14.0 - cy);
                         if (dx == 0 && dy == 0) return;

                         float tmp[28 * 28];
                         shift28(img28, tmp, dx, dy);
                         for (int i = 0; i < 28 * 28; i++) img28[i] = tmp[i];
                     }

                     static void normalize28_inplace(float *img28) {
                         float mx = 0.0f;
                         for (int i = 0; i < 28 * 28; i++) if (img28[i] > mx) mx = img28[i];
                         if (mx > 1e-8f) {
                             float inv = 1.0f / mx;
                             for (int i = 0; i < 28 * 28; i++) img28[i] *= inv;
                         }
                         for (int i = 0; i < 28 * 28; i++) {
                             if (img28[i] < 0.0f) img28[i] = 0.0f;
                             if (img28[i] > 1.0f) img28[i] = 1.0f;
                         }
                     }

                     static void preprocess_to_mnist28(const uint8_t *canvas, int w, int h, float *out784, int invert) {
                         for (int i = 0; i < 784; i++) out784[i] = 0.0f;

                         int xmin, ymin, xmax, ymax;
                         if (!find_bbox(canvas, w, h, 18, &xmin, &ymin, &xmax, &ymax)) return;

                         int bw = xmax - xmin + 1;
                         int bh = ymax - ymin + 1;

                         int side = (bw > bh) ? bw : bh;
                         int pad = side / 5;
                         side += 2 * pad;
                         if (side < 1) side = 1;

                         int cx = xmin + bw / 2;
                         int cy = ymin + bh / 2;

                         int x0 = cx - side / 2;
                         int y0 = cy - side / 2;

                         if (x0 < 0) x0 = 0;
                         if (y0 < 0) y0 = 0;
                         if (x0 + side > w) x0 = w - side;
                         if (y0 + side > h) y0 = h - side;
                         if (x0 < 0) x0 = 0;
                         if (y0 < 0) y0 = 0;

                         if (x0 + side > w) side = w - x0;
                         if (y0 + side > h) side = h - y0;
                         if (side < 1) return;

                         float tmp20[20 * 20];
                         for (int ty = 0; ty < 20; ty++) {
                             for (int tx = 0; tx < 20; tx++) {
                                 float sx = (float)x0 + ((float)tx + 0.5f) * (float)side / 20.0f;
                                 float sy = (float)y0 + ((float)ty + 0.5f) * (float)side / 20.0f;
                                 float v = sample_bilinear_u8(canvas, w, h, sx, sy) / 255.0f;
                                 if (invert) v = 1.0f - v;
                                 tmp20[ty * 20 + tx] = v;
                             }
                         }

                         int ox = 4, oy = 4;
                         for (int ty = 0; ty < 20; ty++) {
                             for (int tx = 0; tx < 20; tx++) {
                                 out784[(oy + ty) * 28 + (ox + tx)] = tmp20[ty * 20 + tx];
                             }
                         }

                         blur28_inplace(out784);
                         center_of_mass_shift28(out784);
                         normalize28_inplace(out784);
                     }

                     static void topk3(const float *p10, int *i1, int *i2, int *i3) {
                         int a = 0, b = 1, c = 2;
                         if (p10[b] > p10[a]) { int t=a; a=b; b=t; }
                         if (p10[c] > p10[a]) { int t=a; a=c; c=t; }
                         if (p10[c] > p10[b]) { int t=b; b=c; c=t; }
                         for (int i = 3; i < 10; i++) {
                             if (p10[i] > p10[a]) { c=b; b=a; a=i; }
                             else if (p10[i] > p10[b]) { c=b; b=i; }
                             else if (p10[i] > p10[c]) { c=i; }
                         }
                         *i1=a; *i2=b; *i3=c;
                     }

                     int gui_run(const char *model_path) {
                         const int INPUT_DIM = 784;
                         const int NUM_CLASSES = 10;
                         const int HIDDEN_DIM = 128;

                         const int CANVAS_W = 280, CANVAS_H = 280;
                         const int WIN_W = 800, WIN_H = 800;

                         float *W1 = (float*)malloc(sizeof(float) * INPUT_DIM * HIDDEN_DIM);
                         float *b1 = (float*)malloc(sizeof(float) * HIDDEN_DIM);
                         float *W2 = (float*)malloc(sizeof(float) * HIDDEN_DIM * NUM_CLASSES);
                         float *b2 = (float*)malloc(sizeof(float) * NUM_CLASSES);

                         if (!W1 || !b1 || !W2 || !b2) { free(W1); free(b1); free(W2); free(b2); return 1; }

                         if (!mlp_load(model_path, W1, b1, INPUT_DIM, HIDDEN_DIM, W2, b2, NUM_CLASSES)) {
                             free(W1); free(b1); free(W2); free(b2);
                             return 2;
                         }

                         if (SDL_Init(SDL_INIT_VIDEO) != 0) {
                             free(W1); free(b1); free(W2); free(b2);
                             return 3;
                         }

                         SDL_Window *window = SDL_CreateWindow(
                             "MLP Digit Classifier (C clear, Enter predict, [ ] brush, R erase, I invert, Esc quit)",
                                                               SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                                               WIN_W, WIN_H, SDL_WINDOW_SHOWN
                         );
                         if (!window) { SDL_Quit(); free(W1); free(b1); free(W2); free(b2); return 4; }

                         SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
                         if (!renderer) { SDL_DestroyWindow(window); SDL_Quit(); free(W1); free(b1); free(W2); free(b2); return 5; }

                         SDL_Texture *tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, CANVAS_W, CANVAS_H);
                         if (!tex) { SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); SDL_Quit(); free(W1); free(b1); free(W2); free(b2); return 6; }

                         uint8_t *canvas = (uint8_t*)calloc((size_t)CANVAS_W * (size_t)CANVAS_H, 1);
                         uint32_t *rgba = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)CANVAS_W * (size_t)CANVAS_H);
                         if (!canvas || !rgba) {
                             free(canvas); free(rgba);
                             SDL_DestroyTexture(tex); SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); SDL_Quit();
                             free(W1); free(b1); free(W2); free(b2);
                             return 7;
                         }

                         int brush_r = 8;
                         int brush_strength = 255;
                         int invert = 0;
                         int erasing = 0;

                         int drawing = 0;
                         int last_cx = -1, last_cy = -1;

                         float img784[784];
                         float z1[128];
                         float a1[128];
                         float logits10[10];
                         uint8_t pred = 0;

                         int top1 = -1, top2 = -1, top3 = -1;
                         float p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;

                         int running = 1;
                         while (running) {
                             SDL_Event e;
                             while (SDL_PollEvent(&e)) {
                                 if (e.type == SDL_QUIT) running = 0;

                                 if (e.type == SDL_KEYDOWN) {
                                     SDL_Keycode key = e.key.keysym.sym;

                                     if (key == SDLK_ESCAPE) running = 0;

                                     if (key == SDLK_c) {
                                         canvas_clear(canvas, CANVAS_W, CANVAS_H);
                                         SDL_SetWindowTitle(window, "MLP Digit Classifier (C clear, Enter predict, [ ] brush, R erase, I invert, Esc quit)");
                                     }

                                     if (key == SDLK_LEFTBRACKET) { if (brush_r > 1) brush_r--; }
                                     if (key == SDLK_RIGHTBRACKET) { if (brush_r < 40) brush_r++; }

                                     if (key == SDLK_r) erasing = !erasing;
                                     if (key == SDLK_i) invert = !invert;

                                     if (key == SDLK_RETURN || key == SDLK_KP_ENTER) {
                                         preprocess_to_mnist28(canvas, CANVAS_W, CANVAS_H, img784, invert);

                                         mlp_forward(img784, 1, INPUT_DIM, W1, b1, HIDDEN_DIM, W2, b2, NUM_CLASSES, z1, a1, logits10);
                                         softmax_rowwise_inplace(logits10, 1, NUM_CLASSES);
                                         argmax_rowwise(logits10, 1, NUM_CLASSES, &pred);

                                         int a,b,c;
                                         topk3(logits10, &a, &b, &c);

                                         top1 = a; top2 = b; top3 = c;
                                         p1 = logits10[a]; p2 = logits10[b]; p3 = logits10[c];

                                         char title[256];
                                         snprintf(title, sizeof(title),
                                                  "Pred %u (%.3f) | top %d:%.2f %d:%.2f %d:%.2f | brush %d | %s | invert %s",
                                                  (unsigned)pred, logits10[pred],
                                                  a, logits10[a], b, logits10[b], c, logits10[c],
                                                  brush_r, erasing ? "ERASE" : "DRAW", invert ? "ON" : "OFF");
                                         SDL_SetWindowTitle(window, title);
                                     }
                                 }

                                 if (e.type == SDL_MOUSEBUTTONDOWN) {
                                     if (e.button.button == SDL_BUTTON_LEFT) {
                                         drawing = 1;
                                         int cx = e.button.x * CANVAS_W / WIN_W;
                                         int cy = e.button.y * CANVAS_H / WIN_H;
                                         canvas_stamp_brush(canvas, CANVAS_W, CANVAS_H, cx, cy, brush_r, brush_strength, erasing);
                                         last_cx = cx; last_cy = cy;
                                     }
                                     if (e.button.button == SDL_BUTTON_RIGHT) {
                                         drawing = 1;
                                         int cx = e.button.x * CANVAS_W / WIN_W;
                                         int cy = e.button.y * CANVAS_H / WIN_H;
                                         canvas_stamp_brush(canvas, CANVAS_W, CANVAS_H, cx, cy, brush_r, brush_strength, 1);
                                         last_cx = cx; last_cy = cy;
                                     }
                                 }

                                 if (e.type == SDL_MOUSEBUTTONUP) {
                                     if (e.button.button == SDL_BUTTON_LEFT || e.button.button == SDL_BUTTON_RIGHT) {
                                         drawing = 0;
                                         last_cx = -1; last_cy = -1;
                                     }
                                 }

                                 if (e.type == SDL_MOUSEMOTION && drawing) {
                                     int cx = e.motion.x * CANVAS_W / WIN_W;
                                     int cy = e.motion.y * CANVAS_H / WIN_H;

                                     int use_erase = erasing;
                                     if ((e.motion.state & SDL_BUTTON_RMASK) != 0) use_erase = 1;

                                     if (last_cx >= 0 && last_cy >= 0) {
                                         int dx = cx - last_cx;
                                         int dy = cy - last_cy;
                                         int steps = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);
                                         if (steps < 1) steps = 1;

                                         for (int s = 1; s <= steps; s++) {
                                             int ix = last_cx + (dx * s) / steps;
                                             int iy = last_cy + (dy * s) / steps;
                                             canvas_stamp_brush(canvas, CANVAS_W, CANVAS_H, ix, iy, brush_r, brush_strength, use_erase);
                                         }
                                     } else {
                                         canvas_stamp_brush(canvas, CANVAS_W, CANVAS_H, cx, cy, brush_r, brush_strength, use_erase);
                                     }

                                     last_cx = cx; last_cy = cy;
                                 }
                             }

                             canvas_to_rgba(canvas, CANVAS_W, CANVAS_H, rgba);
                             SDL_UpdateTexture(tex, NULL, rgba, CANVAS_W * (int)sizeof(uint32_t));

                             SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                             SDL_RenderClear(renderer);

                             SDL_Rect dst = {0, 0, WIN_W, WIN_H};
                             SDL_RenderCopy(renderer, tex, NULL, &dst);

                             SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
                             SDL_SetRenderDrawColor(renderer, 0, 0, 0, 180);

                             int pad = 16;
                             int scale = 10;
                             int line_h = 7 * scale + 10;
                             int box_h = line_h * 3 + pad * 2;
                             SDL_Rect box = { pad, WIN_H - box_h - pad, WIN_W - pad * 2, box_h };
                             SDL_RenderFillRect(renderer, &box);

                             SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

                             if (top1 >= 0) {
                                 char line1[64], line2[64], line3[64];

                                 int pct1 = (int)lroundf(p1 * 100.0f);
                                 int pct2 = (int)lroundf(p2 * 100.0f);
                                 int pct3 = (int)lroundf(p3 * 100.0f);

                                 snprintf(line1, sizeof(line1), "1:%d %d%%", top1, pct1);
                                 snprintf(line2, sizeof(line2), "2:%d %d%%", top2, pct2);
                                 snprintf(line3, sizeof(line3), "3:%d %d%%", top3, pct3);

                                 int tx = box.x + pad;
                                 int ty = box.y + pad;

                                 draw_text5x7(renderer, tx, ty + 0 * line_h, scale, line1);
                                 draw_text5x7(renderer, tx, ty + 1 * line_h, scale, line2);
                                 draw_text5x7(renderer, tx, ty + 2 * line_h, scale, line3);
                             }

                             SDL_RenderPresent(renderer);
                         }

                         free(canvas);
                         free(rgba);

                         SDL_DestroyTexture(tex);
                         SDL_DestroyRenderer(renderer);
                         SDL_DestroyWindow(window);
                         SDL_Quit();

                         free(W1); free(b1); free(W2); free(b2);
                         return 0;
                     }
