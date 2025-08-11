#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <fstream>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class GravityIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    GravityIntegrator(const Properties &props) : Base(props) {
        initialize_table();
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium *medium, Float *aovs,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // Loop state
        Ray3f ray           = Ray3f(ray_);
        Spectrum throughput = 1.0f;
        Spectrum result     = 0.0f;
        Float eta           = 1.0f;
        UInt32 depth        = 0;

        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // TODO: we probably don't need this or we deal with this at the very
        // end
        Interaction3f prev_si = dr::zeros<Interaction3f>();
        Float prev_bsdf_pdf   = 1.0f;
        Bool prev_bsdf_delta  = true;
        BSDFContext bsdf_ctx;

        struct LoopState {
            Ray3f ray;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool active;
            Sampler *sampler;

            DRJIT_STRUCT(LoopState, ray, throughput, result, eta, depth,
                         valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                         active, sampler)
        } ls = { ray,     throughput,    result,
                 eta,     depth,         valid_ray,
                 prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                 active,  sampler };

        dr::tie(ls) = dr::while_loop(
            dr::make_tuple(ls), [](const LoopState &ls) { return ls.active; },
            [this, scene, bsdf_ctx](LoopState &ls) {
                // Map incoming ray from sensor to ray after the effect of
                // gravity
                Float d          = shortest_distance_point_to_ray(ls.ray);
                Ray3f bended_ray = Ray3f(ls.ray);
                bool valid = false;
                effective_outgoing_ray(bended_ray, valid);

                // Use the calculated ray to get the emitter mapping
                if (valid) {
                    SurfaceInteraction3f si = scene->ray_intersect(
                        bended_ray, +RayFlags::All, ls.depth == 0u);

                    if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                        DirectionSample3f ds(scene, si, ls.prev_si);
                        Float em_pdf = 0.0f;

                        if (dr::any_or<true>(!ls.prev_bsdf_delta)) {
                            em_pdf = scene->pdf_emitter_direction(
                                ls.prev_si, ds, !ls.prev_bsdf_delta);
                        }

                        // Compute weight for emitter sample from previous bounce
                        Float mis_bsdf = 1;

                        // Sample the emitter and accumulate the results
                        ls.result = dr::fmadd(
                            ls.throughput,
                            ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.0f) *
                                mis_bsdf,
                            ls.result);
                    } else {
                        ls.result = 0;
                    }
                } else {
                    // If there is an intersection with the sphere, set the
                    // value manually
                    ls.result = 0;
                }

                ls.active = false;
            });

        return { dr::select(ls.valid_ray, ls.result, 0.0f), ls.valid_ray };
    }

    MI_DECLARE_CLASS()
protected:
    /// Important: declare a protected virtual destructor
    // virtual ~GravityIntegrator();
private:
    std::array<Float, 408> slope_yz_table;
    std::array<Float, 408> pos_x_table;
    std::array<Float, 408> pos_y_table;
    std::array<Float, 408> pos_z_table;
    std::array<Float, 408> mom_x_table;
    std::array<Float, 408> mom_y_table;
    std::array<Float, 408> mom_z_table;
    std::array<bool, 408> valid_table;

    static std::vector<std::string> split_string(const std::string& str) {
        std::stringstream ss(str);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.emplace_back(token);
        }
        return tokens;
    }

    void initialize_table() {
        std::string lookup_table = "lookup_table.csv";
        std::string line;

        // read lookup table
        std::ifstream table(lookup_table);
        std::getline(table, line);
        std::size_t index = 0;
        while (std::getline(table, line)) {
            std::vector<std::string> row = split_string(line);
            slope_yz_table.at(index) = std::stof(row.at(0));
            pos_x_table.at(index) = std::stof(row.at(1));
            pos_y_table.at(index) = std::stof(row.at(2));
            pos_z_table.at(index) = std::stof(row.at(3));
            mom_x_table.at(index) = std::stof(row.at(4));
            mom_y_table.at(index) = std::stof(row.at(5));
            mom_z_table.at(index) = std::stof(row.at(6));
            if (std::stoi(row.at(7)) == 1) {
                valid_table.at(index) = true;
            } else {
                valid_table.at(index) = false;
            }
            index = index + 1;
        }
        table.close();
    }

    static std::array<Float, 3> cross_product(const std::array<Float, 3> &a,
                                              const std::array<Float, 3> &b) {
        return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                 a[0] * b[1] - a[1] * b[0] };
    }

    static Float norm(const std::array<Float, 3> &a) {
        return dr::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }

    static Float shortest_distance_point_to_ray(
        const Ray3f &ray, const std::array<Float, 3> &p0 = { 0.0, 0.0, 0.0 }) {

        // Ray info
        std::array<Float, 3> p1 = { ray.o[0], ray.o[1], ray.o[2] };
        std::array<Float, 3> p2 = { ray.o[0] + ray.d[0], ray.o[1] + ray.d[1],
                                    ray.o[2] + ray.d[2] };

        // Calculate the shortest distance from the ray to point p0
        Float d = norm(cross_product(
                      { p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2] },
                      { p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2] })) /
                  norm({ p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] });

        return d;
    }

    static std::array<Float, 3> rotation(const std::string &axis,
                                         const Float &theta,
                                         const std::array<Float, 3> &p) {
        std::array<Float, 3> rotated_p = p;
        if (axis == "x") {
            rotated_p[0] = p[0];
            rotated_p[1] = dr::cos(theta) * p[1] - dr::sin(theta) * p[2];
            rotated_p[2] = dr::sin(theta) * p[1] + dr::cos(theta) * p[2];
        } else if (axis == "y") {
            rotated_p[0] = dr::cos(theta) * p[0] + dr::sin(theta) * p[2];
            rotated_p[1] = p[1];
            rotated_p[2] = -dr::sin(theta) * p[0] + dr::cos(theta) * p[2];
        } else if (axis == "z") {
            rotated_p[0] = dr::cos(theta) * p[0] - dr::sin(theta) * p[1];
            rotated_p[1] = dr::sin(theta) * p[0] + dr::cos(theta) * p[1];
            rotated_p[2] = p[2];
        } else {
            std::string err_msg = "No axis " + axis;
            Log(Error, err_msg.c_str());
        }
        return rotated_p;
    }

    void get_outgoing_ray(Float slope, std::array<Float, 3> &out_pos,
                          std::array<Float, 3> &out_mom, bool &valid) const {

        bool slope_is_negative = false;
        if (dr::any(slope < 0)) {
            slope_is_negative = true;
            slope = -slope;
        }

        // assume slope is positive
        if (dr::any(slope < slope_yz_table.at(0))) {
            valid = false;
        } else if (dr::any(slope >= slope_yz_table.at(slope_yz_table.size() - 1))) {
            valid = true;
        } else {
            std::size_t index = 0;
            for (std::size_t i = 0; i < slope_yz_table.size(); i++) {
                if (dr::any(slope > slope_yz_table[i])) {
                    index = i;
                    valid = valid_table.at(index);
                    break;
                }
            }

            // interpolation
            Float m = slope - slope_yz_table.at(index);
            Float n = slope - slope_yz_table.at(index + 1);
            out_pos[0] = (n * pos_x_table.at(index) + m * pos_x_table.at(index + 1)) / (m + n);
            out_pos[1] = (n * pos_y_table.at(index) + m * pos_y_table.at(index + 1)) / (m + n);
            out_pos[2] = (n * pos_z_table.at(index) + m * pos_z_table.at(index + 1)) / (m + n);
            out_mom[0] = (n * mom_x_table.at(index) + m * mom_x_table.at(index + 1)) / (m + n);
            out_mom[1] = (n * mom_y_table.at(index) + m * mom_y_table.at(index + 1)) / (m + n);
            out_mom[2] = (n * mom_z_table.at(index) + m * mom_z_table.at(index + 1)) / (m + n);

            // deal with slope is negative before return
            if (slope_is_negative) {
                out_pos[2] = -out_pos[2];
                out_mom[2] = -out_mom[2];
            }
        }
    }

    void effective_outgoing_ray(Ray3f &ray, bool &valid) const {
        // This function changes ray.o and ray.d
        // Currently, it is hard-coded.

        // For now, if the shortest distance is less than the radius simply
        // return assume sphere radius = 2
        Float d = shortest_distance_point_to_ray(ray);
        if (dr::any(d < 2.0f)) {
            valid = false;
            return;
        } else {
            std::array<Float, 3> ray_o = {ray.o[0], ray.o[1], ray.o[2]};
            std::array<Float, 3> ray_d = {ray.d[0], ray.d[1], ray.d[2]};

            // step1: rotate ray.d back to yz plane and get the slope_yz
            Float theta_fix = dr::atan(ray.d[0] / ray.d[2]);
            ray_d = rotation("y", -theta_fix, ray_d);
            Float slope = ray_d[2] / ray_d[1];

            // step2: map to the outgoing ray
            get_outgoing_ray(slope, ray_o, ray_d, valid);

            // step3: rotate back
            ray_o = rotation("y", theta_fix, ray_o);
            ray_d = rotation("y", theta_fix, ray_d);

            // map to value
            for (int i = 0; i < 3; i++) {
                ray.o[i] = ray_o[i];
                ray.d[i] = ray_d[i];
            }
            return;
        }
    }
};

/// Implement RTTI data structures
MI_IMPLEMENT_CLASS_VARIANT(GravityIntegrator, SamplingIntegrator)
MI_EXPORT_PLUGIN(GravityIntegrator, "Integrator with ray bended by gravity")
NAMESPACE_END(mitsuba)
