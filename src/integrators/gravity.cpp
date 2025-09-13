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
        // TODO: this doesn't read/write the file in the same folder as bbh.xml
        if (props.has_property("ray_coordinates")) {
            this->ray_coordinates_filename = props.string("ray_coordinates");
            store_ray_coordinate = true;
            create_ray_file();
        } else {
            store_ray_coordinate = false;
        }

        if (props.has_property("project_to")) {
            this->project_to = props.string("project_to");
        }

        if (props.has_property("stride")) {
            this->stride = props.get<int>("stride");
        }

        if (props.has_property("lookup_table")) {
            this->lookup_table_filename = props.string("lookup_table");
        }

        Log(Info, "ray_coordinates: '%s', lookup_table: '%s'",
            this->ray_coordinates_filename, this->lookup_table_filename);

        read_lookup_table();
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

        // write original rays to file
        if (store_ray_coordinate) {
            std::ofstream of;
            of.open(ray_coordinates_filename, std::ios::app);
            of << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << ",";
            of << ray.d[0] << "," << ray.d[1] << "," << ray.d[2] << "\n";
            of.close();
        }

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
                bool valid = effective_outgoing_ray(bended_ray);

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
    std::string ray_coordinates_filename;
    std::string lookup_table_filename = "lookup_table.txt";
    bool has_lookup_table = false;
    bool store_ray_coordinate = false;

    std::string project_to = "xy";
    int stride = -1;
    std::vector<std::array<Float,3>> in_sample_pos;
    std::vector<std::array<Float,3>> out_ray_pos;
    std::vector<std::array<Float,3>> out_ray_dir;

    static std::vector<std::string> split_string(const std::string& str) {
        std::stringstream ss(str);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.emplace_back(token);
        }
        return tokens;
    }

    void create_ray_file() {
        std::ofstream ray_file(ray_coordinates_filename);
        ray_file << "in_pos_x,in_pos_y,in_pos_z,in_dir_x,in_dir_y,in_dir_z\n";
        ray_file.close();
    }

    /**
     * read_lookup_table
     */
    void read_lookup_table() {
        // read lookup table
        std::ifstream table(this->lookup_table_filename);
        if (!table.is_open()) {
            Log(Warn, "No lookup table file '%s', so this run will only dump the ray coordinates.",
                this->lookup_table_filename);
            return;
        }

        std::string line;
        std::getline(table, line);
        while (std::getline(table, line)) {
            if (line.empty()) {
                break;
            }
            std::vector<std::string> row = split_string(line);
            std::array<Float, 3> sample{stof(row.at(0)), stof(row.at(1)), stof(row.at(2))};
            std::array<Float, 3> out_pos{stof(row.at(3)), stof(row.at(4)), stof(row.at(5))};
            std::array<Float, 3> out_dir{stof(row.at(6)), stof(row.at(7)), stof(row.at(8))};
            in_sample_pos.emplace_back(sample);
            out_ray_pos.emplace_back(out_pos);
            out_ray_dir.emplace_back(out_dir);
        }
        table.close();

        // make sure project_to/width/height are set
        if (stride > 0) {
            has_lookup_table = true;
        } else {
            has_lookup_table = false;
        }
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

    /**
     * get_outgoing_ray: this looks up the table and return the
     */
    void get_outgoing_ray(Float slope, std::array<Float, 3> &out_pos,
                          std::array<Float, 3> &out_mom, bool &valid) const {
        // return origin ray if no lookup table
        if (!has_lookup_table) {
            valid = true;
            return;
        }

    }


    bool effective_outgoing_ray(Ray3f &ray) const {
        // This function changes ray.o and ray.d
        // Currently, it is hard-coded.

        // For now, if the shortest distance is less than the radius simply
        // return assume sphere radius = 2
        Float d = shortest_distance_point_to_ray(ray);
        if (dr::any(d < 2.0f)) {
            bool valid = false;
            return valid;
        } else {
            std::array<Float, 3> ray_o = {ray.o[0], ray.o[1], ray.o[2]};
            std::array<Float, 3> ray_d = {ray.d[0], ray.d[1], ray.d[2]};

            // step1: rotate ray.d back to yz plane and get the slope_yz
            Float theta_fix = dr::atan(ray.d[0] / ray.d[2]);
            ray_d = rotation("y", -theta_fix, ray_d);
            Float slope = ray_d[2] / ray_d[1];

            // step2: map to the outgoing ray
            bool valid = true;
            get_outgoing_ray(slope, ray_o, ray_d, valid);

            // step3: rotate back
            ray_o = rotation("y", theta_fix, ray_o);
            ray_d = rotation("y", theta_fix, ray_d);

            // map to value
            for (int i = 0; i < 3; i++) {
                ray.o[i] = ray_o[i];
                ray.d[i] = ray_d[i];
            }
            return valid;
        }
    }
};

/// Implement RTTI data structures
MI_IMPLEMENT_CLASS_VARIANT(GravityIntegrator, SamplingIntegrator)
MI_EXPORT_PLUGIN(GravityIntegrator, "Integrator with ray bended by gravity")
NAMESPACE_END(mitsuba)
