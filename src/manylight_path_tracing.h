#pragma once

#include "scene.h"
#include "pcg.h"

Spectrum path_tracing_ml(const Scene &scene,
                      int x, int y, int spp,/* pixel coordinates */
                      pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        // Hit background. Account for the environment map if needed.
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
    // A path's contribution is 
    // C(v) = W(v0, v1) * G(v0, v1) * f(v0, v1, v2) * 
    //                    G(v1, v2) * f(v1, v2, v3) * 
    //                  ........
    //                  * G(v_{n-1}, v_n) * L(v_{n-1}, v_n)
    // where v is the path vertices, W is the sensor response
    // G is the geometry term, f is the BSDF, L is the emission
    //
    // "sample_primary" importance samples both W and G,
    // and we assume it always has weight 1.

    // current_path_throughput stores the ratio between
    // 1) the path contribution from v0 up to v_{i} (the BSDF f(v_{i-1}, v_i, v_{i+1}) is not included), 
    // where i is where the PathVertex "vertex" lies on, and
    // 2) the probability density for computing the path v from v0 up to v_i,
    // so that we can compute the Monte Carlo estimates C/p. 
    Spectrum current_path_throughput = fromRGB(Vector3{1, 1, 1});
    // eta_scale stores the scale introduced by Snell-Descartes law to the BSDF (eta^2).
    // We use the same Russian roulette strategy as Mitsuba/pbrt-v3
    // and tracking eta_scale and removing it from the
    // path contribution is crucial for many bounces of refraction.
    Real eta_scale = Real(1);

    // We hit a light immediately. 
    // This path has only two vertices and has contribution
    // C = W(v0, v1) * G(v0, v1) * L(v0, v1)
    if (is_light(scene.shapes[vertex.shape_id])) {
        radiance += current_path_throughput *
            emission(vertex, -ray.dir, scene);
    }

    const Material &mat = scene.materials[vertex.material_id];

    // First, we sample a point on the light source.
    // We do this by first picking a light source, then pick a point on it.

    for (auto &light : scene.lights) {
        for (int i = 0; i < spp; i++) {
            Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            Real light_w = next_pcg32_real<Real>(rng);
            Real shape_w = next_pcg32_real<Real>(rng);
             PointAndNormal point_on_light =
                sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);
            // Next, we compute w1*C1/p1. We store C1/p1 in C1.
            Spectrum C1 = make_zero_spectrum();
            // Remember "current_path_throughput" already stores all the path contribution on and before v_i.
            // So we only need to compute G(v_{i}, v_{i+1}) * f(v_{i-1}, v_{i}, v_{i+1}) * L(v_{i}, v_{i+1})
            {
                Real G = 0;
                Vector3 dir_light;

                if (!is_envmap(light)) {
                    dir_light = normalize(point_on_light.position - vertex.position);

                    Ray shadow_ray{vertex.position, dir_light, 
                                get_shadow_epsilon(scene),
                                (1 - get_shadow_epsilon(scene)) *
                                    distance(point_on_light.position, vertex.position)};
                    if (!occluded(scene, shadow_ray)) {

                        G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                            distance_squared(point_on_light.position, vertex.position);
                    }
                } else {

                    dir_light = -point_on_light.normal;
                    Ray shadow_ray{vertex.position, dir_light, 
                                get_shadow_epsilon(scene),
                                infinity<Real>() /* envmaps are infinitely far away */};
                    if (!occluded(scene, shadow_ray)) {
                        // We integrate envmaps using the solid angle measure,
                        // so the geometry term is 1.
                        G = 1;
                    }
                }

                // Before we proceed, we first compute the probability density p1(v1)
                // The probability density for light sampling to sample our point is
                // just the probability of sampling a light times the probability of sampling a point
                Real p1 = pdf_point_on_light(light, point_on_light, vertex.position, scene);

                if (G > 0 && p1 > 0) {
                    // Let's compute f (BSDF) next.
                    Vector3 dir_view = -ray.dir;
                    assert(vertex.material_id >= 0);
                    Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);

                    Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
                    C1 = G * f * L;
                    C1 /= p1;
                }
            }
            radiance += current_path_throughput * C1 / Real(spp);
        }
    
    }
    return radiance;
}