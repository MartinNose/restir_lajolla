#pragma once

#include "scene.h"
#include "pcg.h"
#include "halton.h"

struct Sample {
    int32_t lightid;
    Vector2 light_uv;
    Real shape_w;
    Real w;
    Vector2 r_pos;

    Sample() : lightid(-1), w(0) {};
    Sample(int lightid, Vector2 light_uv, Real shape_w, Real w)
        : lightid(lightid), light_uv(light_uv), shape_w(shape_w), w(w) {};
    Sample(int lightid, Vector2 light_uv, Real shape_w, Real w, Vector2 r_pos)
        : lightid(lightid), light_uv(light_uv), shape_w(shape_w), w(w), r_pos(r_pos) {};
};

struct Reservior {
    Sample y;
    Real w_sum;
    int16_t M;
    Real W;
    Reservior() : w_sum(0), M(0), W(0) {};
};

void clean(Reservior &reservior) {
    reservior.y = Sample(-1, Vector2{0, 0}, 0, 0);
    reservior.w_sum = 0;
    reservior.M = 0;
    reservior.W = 0;
}

bool update_reservior(Reservior &reservior, Sample x, pcg32_state &rng) {
    reservior.w_sum += x.w;
    reservior.M++;
    if (next_pcg32_real<Real>(rng) < x.w / reservior.w_sum) {
        reservior.y = x;
        return true;
    }
    return false;
}

Reservior reservior_sample(std::vector<Sample> S, pcg32_state &rng) {
    Reservior r;
    for (auto &s : S) {
        update_reservior(r, s, rng);
    }
    return r;
}

Reservior combineReservior(const Scene &scene, int x, int y, pcg32_state &rng,
                const Reservior *r_orig, int k, int split_w, int split_h) {
    Reservior r_q = r_orig[y * split_w + x];

    Ray ray = sample_primary(scene.camera, r_q.y.r_pos);
    RayDifferential ray_diff = init_ray_differential(scene.camera.width, scene.camera.height);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        return r_q;
    }
    PathVertex vertex = *vertex_;
    if (is_light(scene.shapes[vertex.shape_id])) {
        return r_q;
    }
    const Material &mat = scene.materials[vertex.material_id];

    Real d_vc = distance(vertex_->position, ray.org);

    Real selected_p_tld = 0;
    bool flag = false;

    Reservior s;
    clean(s);
    int M = 0;

    {
        Real p_tld;
        if (r_q.y.lightid != -1) {
            Light light = scene.lights[r_q.y.lightid];

            PointAndNormal point_on_light =
                    sample_point_on_light(light, vertex.position, r_q.y.light_uv, r_q.y.shape_w, scene);
            Vector3 dir_light = normalize(point_on_light.position - vertex.position);
            Spectrum f = eval(mat, -ray.dir, dir_light, vertex, scene.texture_pool);
            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
            Real G = 0;
            Ray shadow_ray{vertex.position, dir_light, 
                                get_shadow_epsilon(scene),
                                (1 - get_shadow_epsilon(scene)) *
                                    distance(point_on_light.position, vertex.position)};
            bool shadowed = occluded(scene, shadow_ray);
            if (!shadowed) {
                G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                    distance_squared(point_on_light.position, vertex.position);
            }
            p_tld = luminance(f * L) * G;
        } else {
            p_tld = 0;
        }
       
        if (update_reservior(s, Sample(r_q.y.lightid, r_q.y.light_uv, r_q.y.shape_w, p_tld * r_q.W * r_q.M, r_q.y.r_pos), rng)) {
            selected_p_tld = p_tld;
            flag = true;
        }
        M += r_q.M;
    }

    int cnt = 1;
    int attempts = 0;
    while(cnt < k + 1) {
        attempts++;
        if (attempts > k * 3) {
            break;
        }
        Vector2i q(x, y);
        // Vector2 offset = next_halton_sequence2d(hrng) * Real(30);
        Vector2 offset(next_pcg32_real<Real>(rng) * 30, 
            next_pcg32_real<Real>(rng) * 30);
        q.x += floor(offset.x);
        q.y += floor(offset.y);
        if (q.x < 0 || q.x >= split_w ||q.y < 0 ||q.y >= split_h) {
            continue;
        }
        Reservior r = r_orig[q.y * split_w + q.x];
        if (r.M == 0 || r.y.lightid == -1) {
            continue;
        }
        Ray test_ray = sample_primary(scene.camera, r.y.r_pos);
        std::optional<PathVertex> test_vertex_ = intersect(scene, test_ray, ray_diff);
        if (!test_vertex_) {
            continue;
        }
        Real d_vt = distance(test_vertex_->position, test_ray.org);
        if (fabs(d_vt - d_vc) > 0.1 * d_vc) {
            continue;
        }

        Real theta = acos(dot(normalize(vertex.geometric_normal), normalize(test_vertex_->geometric_normal)));
        if (theta > (Real(25)/Real(180)) * M_PI) {
            continue;
        }

        Sample c = r.y;
        if (r.M == 0) {
            continue;
        }
        Real p_tld;
        if (c.lightid == -1) {
            p_tld = 0;
        } else {
            Light light = scene.lights[c.lightid];

            PointAndNormal point_on_light =
                    sample_point_on_light(light, vertex.position, c.light_uv, c.shape_w, scene);
            Vector3 dir_light = normalize(point_on_light.position - vertex.position);
            Spectrum f = eval(mat, -ray.dir, dir_light, vertex, scene.texture_pool);
            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
            Real G = 0;
            Ray shadow_ray{vertex.position, dir_light, 
                                get_shadow_epsilon(scene),
                                (1 - get_shadow_epsilon(scene)) *
                                    distance(point_on_light.position, vertex.position)};
            bool shadowed = occluded(scene, shadow_ray);
            if (!shadowed) {
                G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                    distance_squared(point_on_light.position, vertex.position);
            }
            p_tld = luminance(f * L) * G;
        }
        if (update_reservior(s, Sample(c.lightid, c.light_uv, c.shape_w, p_tld * r.W * r.M, r_q.y.r_pos), rng)) {
            selected_p_tld = p_tld;
            flag = true;
        }
        M += r.M;
        cnt++;
    }

    s.M = M;
    if (flag) {
        s.W = 1 / selected_p_tld * (1 / Real(s.M) * s.w_sum);
    } else {
        s = r_q;
    }

    return s;
}


Spectrum shade_pixel(const Scene &scene, const Reservior &r) {
    int w = scene.camera.width, h = scene.camera.height;
    Ray ray = sample_primary(scene.camera, r.y.r_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        if (has_envmap(scene)) {
            const Light &envmap = get_envmap(scene);
            return emission(envmap,
                            -ray.dir, // pointing outwards from light
                            ray_diff.spread,
                            PointAndNormal{}, // dummy parameter for envmap
                            scene);
        }
        return make_zero_spectrum();
    }
    Spectrum radiance = make_zero_spectrum();

    PathVertex vertex = *vertex_;
    if (is_light(scene.shapes[vertex.shape_id])) {
        radiance += emission(vertex, -ray.dir, scene);
    }
    if (r.y.lightid == -1 && r.y.w == 0) {
        return radiance;
    }

    Light light = scene.lights[r.y.lightid];

    
    PointAndNormal point_on_light =
            sample_point_on_light(light, vertex.position, r.y.light_uv, r.y.shape_w, scene);

    const Material &mat = scene.materials[vertex.material_id];
    Spectrum C1 = make_zero_spectrum();

    {
        Real G = 0;
        Vector3 dir_light;
        dir_light = normalize(point_on_light.position - vertex.position);

        Ray shadow_ray{vertex.position, dir_light, 
                    get_shadow_epsilon(scene),
                    (1 - get_shadow_epsilon(scene)) *
                        distance(point_on_light.position, vertex.position)};
        if (!occluded(scene, shadow_ray)) {
            G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, vertex.position);
        }

        Real p1 = pdf_point_on_light(light, point_on_light, vertex.position, scene);
        Real w1 = 0;
        if (G > 0 && p1 > 0) {
            Vector3 dir_view = -ray.dir;
            assert(vertex.material_id >= 0);
            Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);

            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
            C1 = G * f * L;
            if (isnan(r.W)) {
                return radiance;
            } else {
                C1 *= r.W;
                radiance += C1;
            }
        }
    }
    return radiance;
}

void RIS(const Scene &scene, int x, int y, pcg32_state &rng, Reservior &reservior) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    reservior.y.r_pos = screen_pos; // In case no sample is selected in the reservior
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        reservior.y = Sample(-1, Vector2{0, 0}, 0, 0, screen_pos);
        return;
    }
    PathVertex vertex = *vertex_;

    Spectrum radiance = make_zero_spectrum();

    const Material &mat = scene.materials[vertex.material_id];
    bool isshadow = false;
    Real select_p_tld = 0;
    bool flag = false;
    for (int i = 0; i < scene.options.rsvr_size || (!flag && i < 2 * scene.options.rsvr_size); i++) {
        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        PointAndNormal point_on_light =
            sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);
        Real p = light_pmf(scene, light_id) * 
                pdf_point_on_light(light, point_on_light, vertex.position, scene);
        Vector3 dir_view = -ray.dir;
        Vector3 dir_light = normalize(point_on_light.position - vertex.position);
        Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);
        Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
        Real G = 0;
        Ray shadow_ray{vertex.position, dir_light, 
                               get_shadow_epsilon(scene),
                               (1 - get_shadow_epsilon(scene)) *
                                   distance(point_on_light.position, vertex.position)};
        bool shadowed = occluded(scene, shadow_ray);
        if (!shadowed) {
            G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, vertex.position);
        }
        Real p_tld = luminance(f * L) * G;
        Real w = p_tld / p;
        Sample x = {light_id, light_uv, shape_w, w, screen_pos};
        if(update_reservior(reservior, x, rng)) {
            isshadow = shadowed;
            select_p_tld = p_tld;
            flag = true;
        }
    }
    if (!isshadow) {
        reservior.W = 1/ select_p_tld * (1 / Real(reservior.M) * reservior.w_sum);
    } else {
        reservior.W = 0;
    }
}

Spectrum path_tracing_restir(const Scene &scene,
                      int x, int y, /* pixel coordinates */
                      pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        if (has_envmap(scene)) {
            const Light &envmap = get_envmap(scene);
            return emission(envmap,
                            -ray.dir, // pointing outwards from light
                            ray_diff.spread,
                            PointAndNormal{}, // dummy parameter for envmap
                            scene);
        }
        return make_zero_spectrum();
    }
    PathVertex vertex = *vertex_;

    Spectrum radiance = make_zero_spectrum();
    Real eta_scale = Real(1);

    if (is_light(scene.shapes[vertex.shape_id])) {
        radiance += emission(vertex, -ray.dir, scene);
    }

    const Material &mat = scene.materials[vertex.material_id];

    Reservior rsvr;
    Real selected_p_tld;
    for (int i = 0; i < scene.options.rsvr_size; i++) {
        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        PointAndNormal point_on_light =
            sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);
        Real p = light_pmf(scene, light_id) * 
                pdf_point_on_light(light, point_on_light, vertex.position, scene);
        Vector3 dir_view = -ray.dir;
        Vector3 dir_light = normalize(point_on_light.position - vertex.position);
        Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);
        Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
        Real G = 0;
        Ray shadow_ray{vertex.position, dir_light, 
                               get_shadow_epsilon(scene),
                               (1 - get_shadow_epsilon(scene)) *
                                   distance(point_on_light.position, vertex.position)};
        if (!occluded(scene, shadow_ray)) {
            G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, vertex.position);
        }
        Real p_tld = luminance(f * L) * G;
        Sample x = {light_id, light_uv, shape_w, p_tld / p};
        if (update_reservior(rsvr, x, rng)) {
            selected_p_tld = p_tld;
        }
    }
    rsvr.W = 1/ selected_p_tld * (1 / Real(rsvr.M) * rsvr.w_sum);

    Light light = scene.lights[rsvr.y.lightid];
    PointAndNormal point_on_light =
            sample_point_on_light(light, vertex.position, rsvr.y.light_uv, rsvr.y.shape_w, scene);
    Spectrum C1 = make_zero_spectrum();
    {
        Real G = 0;
        Vector3 dir_light;
        dir_light = normalize(point_on_light.position - vertex.position);

        Ray shadow_ray{vertex.position, dir_light, 
                    get_shadow_epsilon(scene),
                    (1 - get_shadow_epsilon(scene)) *
                        distance(point_on_light.position, vertex.position)};
        if (!occluded(scene, shadow_ray)) {
            G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, vertex.position);
        }

        Real p1 = pdf_point_on_light(light, point_on_light, vertex.position, scene);
        if (G > 0 && p1 > 0) {
            Vector3 dir_view = -ray.dir;
            assert(vertex.material_id >= 0);
            Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);

            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
            C1 = G * f * L;
            if (isnan(rsvr.W)) {
                C1 = make_zero_spectrum();
            } else {
                C1 *= rsvr.W;
            }

            radiance += C1;
        // Real p2 = pdf_sample_bsdf(
        //         mat, dir_view, dir_light, vertex, scene.texture_pool);
        // Real p1 = 1/rsvr.W;
        // w1 = (p1*p1) / (p1*p1 + p2*p2);
        //     if (isnan(p1) || isnan(p2) || isnan(w1)) {
        //         radiance += C1;
        //         return radiance;
        //     }
        // }
        // radiance += C1;
        }
    }

    // {
    //     Vector3 dir_view = -ray.dir;
    //     Vector2 bsdf_rnd_param_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    //     Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
    //     std::optional<BSDFSampleRecord> bsdf_sample_ =
    //         sample_bsdf(mat,
    //                     dir_view,
    //                     vertex,
    //                     scene.texture_pool,
    //                     bsdf_rnd_param_uv,
    //                     bsdf_rnd_param_w);
    //     if (!bsdf_sample_) {
    //         // BSDF sampling failed. Abort the loop.
    //         return radiance; 
    //     }
    //     const BSDFSampleRecord &bsdf_sample = *bsdf_sample_;
    //     Vector3 dir_bsdf = bsdf_sample.dir_out;
    //     // Update ray differentials & eta_scale
    //     if (bsdf_sample.eta == 0) {
    //         ray_diff.spread = reflect(ray_diff, vertex.mean_curvature, bsdf_sample.roughness);
    //     } else {
    //         ray_diff.spread = refract(ray_diff, vertex.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
    //     }

    //     // Trace a ray towards bsdf_dir. Note that again we have
    //     // to have an "epsilon" tnear to prevent self intersection.
    //     Ray bsdf_ray{vertex.position, dir_bsdf, get_intersection_epsilon(scene), infinity<Real>()};
    //     std::optional<PathVertex> bsdf_vertex = intersect(scene, bsdf_ray);

    //     // To update current_path_throughput
    //     // we need to multiply G(v_{i}, v_{i+1}) * f(v_{i-1}, v_{i}, v_{i+1}) to it
    //     // and divide it with the pdf for getting v_{i+1} using hemisphere sampling.
    //     Real G;
    //     if (bsdf_vertex) {
    //         G = fabs(dot(dir_bsdf, bsdf_vertex->geometric_normal)) /
    //             distance_squared(bsdf_vertex->position, vertex.position);
    //     } else {
    //         // We hit nothing, set G to 1 to account for the environment map contribution.
    //         G = 1;
    //     }

    //     Spectrum f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
    //     Real p2 = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);


    //     // Remember to convert p2 to area measure!
    //     p2 *= G;
    //     // note that G cancels out in the division f/p, but we still need
    //     // G later for the calculation of w2.

    //     // Now we want to check whether dir_bsdf hit a light source, and
    //     // account for the light contribution (C2 & w2 & p2).
    //     // There are two possibilities: either we hit an emissive surface,
    //     // or we hit an environment map.
    //     // We will handle them separately.
    //     if (bsdf_vertex && is_light(scene.shapes[bsdf_vertex->shape_id])) {
    //         // G & f are already computed.
    //         Spectrum L = emission(*bsdf_vertex, -dir_bsdf, scene);
    //         Spectrum C2 = G * f * L;
    //         // Next let's compute p1(v2): the probability of the light source sampling
    //         // directly drawing the point corresponds to bsdf_dir.
    //         Real p1_tld = luminance(L * f) * G;
    //         Real p1 = 1 / p1_tld * (1 / Real(rsvr.M) * rsvr.w_sum);
    //         p1 = 1 / p1;
    //         Real w2 = (p2*p2) / (p1*p1 + p2*p2);

    //         C2 /= p2;
    //         radiance += C2 * w2;
    //     }
    // }
    return radiance;
}