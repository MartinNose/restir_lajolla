#include "render.h"
#include "intersection.h"
#include "material.h"
#include "parallel.h"
#include "path_tracing.h"
#include "vol_path_tracing.h"
#include "manylight_path_tracing.h"
#include "reSTIR.h"
#include "pcg.h"
#include "halton.h"
#include "progress_reporter.h"
#include "scene.h"
#include <chrono>

/// Render auxiliary buffers e.g., depth.
Image3 aux_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    parallel_for([&](const Vector2i &tile) {
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Ray ray = sample_primary(scene.camera, Vector2((x + Real(0.5)) / w, (y + Real(0.5)) / h));
                RayDifferential ray_diff = init_ray_differential(w, h);
                if (std::optional<PathVertex> vertex = intersect(scene, ray, ray_diff)) {
                    Real dist = distance(vertex->position, ray.org);
                    Vector3 color{0, 0, 0};
                    if (scene.options.integrator == Integrator::Depth) {
                        color = Vector3{dist, dist, dist};
                    } else if (scene.options.integrator == Integrator::ShadingNormal) {
                        // color = (vertex->shading_frame.n + Vector3{1, 1, 1}) / Real(2);
                        color = vertex->shading_frame.n;
                    } else if (scene.options.integrator == Integrator::MeanCurvature) {
                        Real kappa = vertex->mean_curvature;
                        color = Vector3{kappa, kappa, kappa};
                    } else if (scene.options.integrator == Integrator::RayDifferential) {
                        color = Vector3{ray_diff.radius, ray_diff.spread, Real(0)};
                    } else if (scene.options.integrator == Integrator::MipmapLevel) {
                        const Material &mat = scene.materials[vertex->material_id];
                        const TextureSpectrum &texture = get_texture(mat);
                        auto *t = std::get_if<ImageTexture<Spectrum>>(&texture);
                        if (t != nullptr) {
                            const Mipmap3 &mipmap = get_img3(scene.texture_pool, t->texture_id);
                            Vector2 uv{modulo(vertex->uv[0] * t->uscale, Real(1)),
                                       modulo(vertex->uv[1] * t->vscale, Real(1))};
                            // ray_diff.radius stores approximatedly dpdx,
                            // but we want dudx -- we get it through
                            // dpdx / dpdu
                            Real footprint = vertex->uv_screen_size;
                            Real scaled_footprint = max(get_width(mipmap), get_height(mipmap)) *
                                                    max(t->uscale, t->vscale) * footprint;
                            Real level = log2(max(scaled_footprint, Real(1e-8f)));
                            color = Vector3{level, level, level};
                        }
                    }
                    img(x, y) = color;
                } else {
                    img(x, y) = Vector3{0, 0, 0};
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}

Image3 path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                for (int s = 0; s < spp; s++) {
                    radiance += path_tracing(scene, x, y, rng);
                }
                img(x, y) = radiance / Real(spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    return img;
}

Image3 vol_path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    auto f = vol_path_tracing;
    if (scene.options.vol_path_version == 1) {
        f = vol_path_tracing_1;
    } else if (scene.options.vol_path_version == 2) {
        f = vol_path_tracing_2;
    } else if (scene.options.vol_path_version == 3) {
        f = vol_path_tracing_3;
    } else if (scene.options.vol_path_version == 4) {
        f = vol_path_tracing_4;
    } else if (scene.options.vol_path_version == 5) {
        f = vol_path_tracing_5;
    } else if (scene.options.vol_path_version == 6) {
        f = vol_path_tracing;
    }

    std::chrono::microseconds max = std::chrono::microseconds(0);
    // ProgressReporter reporter(num_tiles_x * num_tiles_y);
    ProgressReporter reporter(w * h);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                // if (!(x == 107 && y == 426)) continue;
                // std:: cout << x << " " << y << std::endl;
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                
                for (int s = 0; s < spp; s++) {
                    auto start = std::chrono::high_resolution_clock::now();
                    Spectrum L = f(scene, x, y, rng);
                    auto end = std::chrono::high_resolution_clock::now(); 
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                    if (duration > max) {
                        max = duration;
                    }
                    // std::cerr << "\r " << x << " " << y << " " << s ;
                    // std::cerr << " " << duration.count();
                    if (isfinite(L)) {
                        // Hacky: exclude NaNs in the rendering.
                        radiance += L;
                    }
                }
                img(x, y) = radiance / Real(spp);
                reporter.update(1);
            }
        }
        // reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    return img;
}

Image3 many_light_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                // if (!(x == 258 && y == 819)) continue;
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                radiance += path_tracing_ml(scene, x, y, spp, rng);
                img(x, y) = radiance;
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    return img;
}

Image3 restir_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

  
    std::cout << "init reserviors done" << std::endl;
    constexpr int split_w = 300;
    constexpr int split_h = 300;

    Reservior r_buffer_1[90000];
    Reservior r_buffer_2[90000];

    Reservior *reserviors = r_buffer_1;
    int num_splits_x = (w + split_w - 1) / split_w;
    int num_splits_y = (h + split_h - 1) / split_h;

    constexpr int tile_size = 16;
    int num_tiles_x = (split_w + tile_size - 1) / tile_size;
    int num_tiles_y = (split_h + tile_size - 1) / tile_size;
    int N = scene.options.samples_per_pixel;
    ProgressReporter reporter((2 + scene.options.sp_reuse_iterations) * num_splits_x * num_splits_y * num_tiles_x * num_tiles_y * N);

    for (int iter = 0; iter < N; iter++) {
    for (int sp_j = 0; sp_j < num_splits_y; sp_j++) {
        for (int sp_i = 0; sp_i < num_splits_x; sp_i++) {
            int achor_x = sp_i * split_w;
            int achor_y = sp_j * split_h;
            reserviors = r_buffer_1;
            Reservior *new_res = r_buffer_2;
            parallel_for([&](const Vector2i &tile) {
                // Use a different rng stream for each thread.
                pcg32_state rng = init_pcg32((sp_j * num_tiles_y + tile[1]) * (w / tile_size)
                                                + sp_i * num_tiles_x + tile[0]);

                int x0 = sp_i * split_w + tile[0] * tile_size;
                int x1 = min(x0 + tile_size, min(w, achor_x + split_w));
                int y0 = sp_j * split_h + tile[1] * tile_size;
                int y1 = min(y0 + tile_size, min(h, achor_y + split_h));
                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        clean(reserviors[(y - achor_y) * split_w  + (x-achor_x)]);        
                        RIS(scene, x, y, rng, reserviors[(y - achor_y) * split_w  + (x-achor_x)]);
                    }
                }
                reporter.update(1);
            }, Vector2i(num_tiles_x, num_tiles_y));

            // Spatial reuse
            if (scene.options.nbr_slt > 0) {
            
            for (int n = 0; n < scene.options.sp_reuse_iterations; n++) {
                memset(new_res, 0, sizeof(Reservior) * split_w * split_h);
                parallel_for([&](const Vector2i &tile) {
                    // Use a different rng stream for each thread.
                    pcg32_state rng = init_pcg32((num_splits_x * num_splits_y * num_tiles_x * num_tiles_y * (1 + n) 
                                                    + (sp_j * num_tiles_y + tile[1]) * (w / tile_size) + sp_i * num_tiles_x + tile[0] * 2));
                    // halton_state hrng = init_halton_state(n * num_tiles_x * num_tiles_y * num_splits_x * num_splits_y + ((sp_j * num_tiles_y + tile[1]) * (w / tile_size)
                                                    // + sp_i * num_tiles_x + tile[0]));
                    int x0 = sp_i * split_w + tile[0] * tile_size;
                    int x1 = min(x0 + tile_size, min(w, achor_x + split_w));
                    int y0 = sp_j * split_h + tile[1] * tile_size;
                    int y1 = min(y0 + tile_size, min(h, achor_y + split_h));
                    for (int y = y0; y < y1; y++) {
                        for (int x = x0; x < x1; x++) {
                            new_res[(y - achor_y) * split_w  + (x-achor_x)] = 
                                combineReservior(scene, x - achor_x, y - achor_y,
                                     rng, reserviors, scene.options.nbr_slt, split_w, split_h);
                        }
                    }
                    reporter.update(1);
                }, Vector2i(num_tiles_x, num_tiles_y));

                if (n % 2 == 0) {
                    reserviors = r_buffer_2;
                    new_res = r_buffer_1;
                } else if (n % 2 == 1) {
                    reserviors = r_buffer_1;
                    new_res = r_buffer_2;
                }
            }}

            parallel_for([&](const Vector2i &tile) {
                // Use a different rng stream for each thread.
                int x0 = sp_i * split_w + tile[0] * tile_size;
                int x1 = min(x0 + tile_size, min(w, achor_x + split_w));
                int y0 = sp_j * split_h + tile[1] * tile_size;
                int y1 = min(y0 + tile_size, min(h, achor_y + split_h));
                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        img(x, y) = shade_pixel(scene, reserviors[(y - achor_y) * split_w + (x - achor_x)]) / Real(N);
                    }
                }
                reporter.update(1);
            }, Vector2i(num_tiles_x, num_tiles_y));
        
        // parallel_for([&](const Vector2i &tile) {
            //     // Use a different rng stream for each thread.
            //     pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
            //     int x0 = tile[0] * tile_size;
            //     int x1 = min(x0 + tile_size, w);
            //     int y0 = tile[1] * tile_size;
            //     int y1 = min(y0 + tile_size, h);
            //     for (int y = y0; y < y1; y++) {
            //         for (int x = x0; x < x1; x++) {
            //             RIS(scene, x, y, rng, reserviors[y * w + x]);
            //         }
            //     }
            // }, Vector2i(num_tiles_x, num_tiles_y));

            // for (int i = 0; i < scene.options.sp_reuse_iterations; i++) {
            //     parallel_for([&](const Vector2i &tile) {
            //         // Use a different rng stream for each thread.
            //         pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
            //         int x0 = tile[0] * tile_size;
            //         int x1 = min(x0 + tile_size, w);
            //         int y0 = tile[1] * tile_size;
            //         int y1 = min(y0 + tile_size, h);
            //         for (int y = y0; y < y1; y++) {
            //             for (int x = x0; x < x1; x++) {
            //                 for (int j = 0; j < scene.options.nbr_slt; j++) {
            //                     Reservior rj = reserviors[y * w + x];
            //                     update_reservior(reserviors[y * w + x], 
            //                         rj.y ,rng);
            //                 }
            //             }
            //         }
            //     }, Vector2i(num_tiles_x, num_tiles_y));
        // }

            // parallel_for([&](const Vector2i &tile) {
            //     // Use a different rng stream for each thread.
            //     pcg32_state rng = init_pcg32((sp_j * num_tiles_y + tile[1]) * (w / tile_size)
            //                                     + sp_i * num_tiles_x + tile[0]);

            //     int x0 = sp_i * split_w + tile[0] * tile_size;
            //     int x1 = min(x0 + tile_size, min(w, achor_x + split_w));
            //     int y0 = sp_j * split_h + tile[1] * tile_size;
            //     int y1 = min(y0 + tile_size, min(h, achor_y + split_h));
            //     for (int y = y0; y < y1; y++) {
            //         for (int x = x0; x < x1; x++) {
            //             Spectrum radiance = make_zero_spectrum();
            //             // radiance += shade_pixel(scene, reserviors[y * w + x]);
            //             radiance += path_tracing_restir(scene, x, y, rng);
            //             img(x, y) += radiance / Real(N);
            //         }
            //     }
            //     reporter.update(1);
            // }, Vector2i(num_tiles_x, num_tiles_y));
        }
    }}
    reporter.done();
    return img;
}


Image3 render(const Scene &scene) {
    if (scene.options.integrator == Integrator::Depth ||
            scene.options.integrator == Integrator::ShadingNormal ||
            scene.options.integrator == Integrator::MeanCurvature ||
            scene.options.integrator == Integrator::RayDifferential ||
            scene.options.integrator == Integrator::MipmapLevel) {
        return aux_render(scene);
    } else if (scene.options.integrator == Integrator::Path) {
        return path_render(scene);
    } else if (scene.options.integrator == Integrator::VolPath) {
        return vol_path_render(scene);
    } else if (scene.options.integrator == Integrator::ManyLight) {
        return many_light_render(scene);
    } else if (scene.options.integrator == Integrator::ReSTIR) {
        return restir_render(scene);
    } else {
        assert(false);
        return Image3();
    }
}
