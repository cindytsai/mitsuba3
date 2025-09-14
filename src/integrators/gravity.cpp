#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <fstream>
#include <algorithm>

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
            std::string project_to = props.string("project_to");
            for (int a = 0; a < 2; a++) {
                if (project_to.at(a) == 'x') {
                    project_axis[a] = 0;
                } else if (project_to.at(a) == 'y') {
                    project_axis[a] = 1;
                } else if (project_to.at(a) == 'z') {
                    project_axis[a] = 2;
                }
            }
        }

        if (props.has_property("stride")) {
            this->stride = props.get<int>("stride");
        }

        if (props.has_property("lookup_table")) {
            this->lookup_table_filename = props.string("lookup_table");
        }

        if (props.has_property("n1")) {
            this->sample_plane[0] = props.get<Float>("n1");
        }

        if (props.has_property("n2")) {
            this->sample_plane[1] = props.get<Float>("n2");
        }

        if (props.has_property("n3")) {
            this->sample_plane[2] = props.get<Float>("n3");
        }

        if (props.has_property("m")) {
            this->sample_plane[3] = props.get<Float>("m");
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

    int project_axis[2] = {0, 1};
    int stride = -1;
    std::array<Float, 4> sample_plane = {0, 0, 0, 0};
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
     * This looks up the table and return the out going ray.
     * @param sample the sample point where the incoming ray intersects the sample plane
     * @param out_pos outgoing ray position
     * @param out_dir outgoing ray direction
     * @param valid validity of the outgoing ray
     */
    void get_outgoing_ray(const std::array<Float, 3>& sample, std::array<Float, 3> &out_pos,
                          std::array<Float, 3> &out_dir, bool &valid) const {
        // return origin ray if no lookup table
        if (!has_lookup_table) {
            valid = true;
            return;
        }

        // find the four neighboring sample points in the table
        // if unfortunate the index exceed the map range, then early return
        std::array<int, 4> neighbor_index = find_neighbor_index(sample);
        for (int i = 0; i < 4; i++) {
            if (neighbor_index[i] < 0 || neighbor_index[i] >= (int) in_sample_pos.size()) {
                return;
            }
        }

        // interpolation (works for only rectangular grids)
        std::array<Float, 3> p1 = in_sample_pos[neighbor_index[0]];
        std::array<Float, 3> p2 = in_sample_pos[neighbor_index[1]];
        std::array<Float, 3> p3 = in_sample_pos[neighbor_index[2]];

        Float f = dr::abs(sample[project_axis[0]] - p1[project_axis[0]]);
        Float g = dr::abs(p2[project_axis[0]] - sample[project_axis[0]]);
        Float m = dr::abs(p3[project_axis[1]] - sample[project_axis[1]]);
        Float n = dr::abs(sample[project_axis[1]] - p1[project_axis[1]]);

        std::array<Float, 3> p1_ray_pos = out_ray_pos[neighbor_index[0]];
        std::array<Float, 3> p2_ray_pos = out_ray_pos[neighbor_index[1]];
        std::array<Float, 3> p3_ray_pos = out_ray_pos[neighbor_index[2]];
        std::array<Float, 3> p4_ray_pos = out_ray_pos[neighbor_index[3]];

        std::array<Float, 3> p1_ray_dir = out_ray_dir[neighbor_index[0]];
        std::array<Float, 3> p2_ray_dir = out_ray_dir[neighbor_index[1]];
        std::array<Float, 3> p3_ray_dir = out_ray_dir[neighbor_index[2]];
        std::array<Float, 3> p4_ray_dir = out_ray_dir[neighbor_index[3]];

        for (int i = 0; i < 3; i++) {
            out_pos[i] = (m/(n+m)) * (g * p1_ray_pos[i] + f * p2_ray_pos[i]) / (g+f) + \
                         (n/(n+m)) * (g * p3_ray_pos[i] + f * p4_ray_pos[i]) / (g+f);
            out_dir[i] = (m/(n+m)) * (g * p1_ray_dir[i] + f * p2_ray_dir[i]) / (g+f) + \
                         (n/(n+m)) * (g * p3_ray_dir[i] + f * p4_ray_dir[i]) / (g+f);
        }
    }

    /**
     * Return neighboring sample points index, from small to large index
     * Assume every ray is within the map
     * *p1 ---- *p2
     *  |        |
     *  |  x     |
     *  |        |
     * *p3 ---- *p4
     *
     * @return p1, p2, p3, p4 index, index must within the range of the table
     */
    std::array<int, 4> find_neighbor_index(const std::array<Float, 3>& sample) const {

        std::array<int, 4> neighbor_index = {-1, -1, -1, -1};

        // find nearest point where the ray sample closest to the table sample
        Float min_value = distance(in_sample_pos[0], in_sample_pos[in_sample_pos.size() - 1]);
        std::size_t min_index = -1;
        for (std::size_t s = 0; s < in_sample_pos.size(); s++) {
            Float d = distance(sample, in_sample_pos[s]);
            if (dr::any(d < min_value)) {
                min_value = d;
                min_index = s;
            }
        }

        // if index out of range then we pad it
        Float w = distance(in_sample_pos[0], in_sample_pos[1]);
        Float h = distance(in_sample_pos[0], in_sample_pos[stride]);
        Float e[4] = {w, h, w, h};
        int index_e[4] = {
            (int) min_index + 1,
            (int) min_index - stride,
            (int) min_index - 1,
            (int) min_index + stride
        };
        for (int i = 0; i < 5; i++) {
            if (index_e[i] >= 0 && index_e[i] < (int) in_sample_pos.size()) {
                e[i] = distance(sample, in_sample_pos[index_e[i]]);
            }
        }

        int shift1 = 0, shift2 = 0;
        if (dr::any(e[2] < e[0])) {
            shift1 = -1;
        } else {
            shift1 = 1;
        }
        if (dr::any(e[3] < e[1])) {
            shift2 = stride;
        } else {
            shift2 = -stride;
        }

        // assigning neighbor index, sort the index from small to large
        neighbor_index[0] = (int) min_index;
        neighbor_index[1] = (int) min_index + shift1;
        neighbor_index[2] = (int) min_index + shift2;
        neighbor_index[3] = (int) min_index + shift1 + shift2;
        std::sort(std::begin(neighbor_index), std::end(neighbor_index));

        return neighbor_index;
    }

    /**
     * Calculate the distance between the point
     * @param p1 (x,y,z) position of the point
     * @param p2 (x,y,z) position of the point
     * @return distance between the point
     */
    Float distance(const std::array<Float, 3> &p1, const std::array<Float, 3> &p2) const {
        std::array<Float, 3> v = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; i++) {
            v[i] = p2[i] - p1[i];
        }

        Float distance = dr::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

        return distance;
    }


    /**
     * Calculate where ray intersect with the sample plane. (n1.x+n2.y+n3.z=m)
     * The sample plane must be in the way of the trajectory of the rays,
     * which is t > 0.
     *
     * @param sample point where the ray intersects with the sample plane
     * @param ray_o ray origin
     * @param ray_d ray direction
     */
    void project_to_sample_plane(std::array<Float, 3>& sample, std::array<Float, 3> ray_o, std::array<Float, 3> ray_d) const {
        Float t = -(sample_plane[0] * ray_o[0] + sample_plane[1] * ray_o[1] + sample_plane[2] * ray_o[2] - sample_plane[3]) / (sample_plane[0] * ray_d[0] + sample_plane[1] * ray_d[1] + sample_plane[2] * ray_d[2]);
        assert(("Sample plane not in front of the ray.", dr::any(t > 0)));

        for (int i = 0; i < 3; i++) {
            sample[i] = ray_o[i] + t * ray_d[i];
        }
    }

    /**
     * Get the outgoing ray after the effect of gravity
     * @param ray outgoing ray (position and direction)
     * @return validity of the ray
     */
    bool effective_outgoing_ray(Ray3f &ray) const {
        // project ray onto sample plane
        std::array<Float, 3> sample = {0.0, 0.0, 0.0};
        std::array<Float, 3> ray_o = {ray.o[0], ray.o[1], ray.o[2]};
        std::array<Float, 3> ray_d = {ray.d[0], ray.d[1], ray.d[2]};
        project_to_sample_plane(sample, ray_o, ray_d);

        // get the outgoing ray through lookup map
        bool valid = true;
        get_outgoing_ray(sample, ray_o, ray_d, valid);

        // update value
        for (int i = 0; i < 3; i++) {
            ray.o[i] = ray_o[i];
            ray.d[i] = ray_d[i];
        }

        return valid;
    }
};

/// Implement RTTI data structures
MI_IMPLEMENT_CLASS_VARIANT(GravityIntegrator, SamplingIntegrator)
MI_EXPORT_PLUGIN(GravityIntegrator, "Integrator with ray bended by gravity")
NAMESPACE_END(mitsuba)
