#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class GravityIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator)
    MI_IMPORT_TYPES(Scene, Sampler, Medium)

    GravityIntegrator(const Properties &props) : Base(props) { }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium *medium,
                                     Float *aovs,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        PreliminaryIntersection3f pi = scene->ray_intersect_preliminary(
            ray, /* coherent = */ true, active);

        return {
            dr::select(pi.is_valid(), pi.t, 0.f),
            pi.is_valid()
        };
    }

    MI_DECLARE_CLASS()
protected:
    /// Important: declare a protected virtual destructor
    // virtual ~GravityIntegrator();
};

/// Implement RTTI data structures
MI_IMPLEMENT_CLASS_VARIANT(GravityIntegrator, SamplingIntegrator)
MI_EXPORT_PLUGIN(GravityIntegrator, "Integrator with ray bended by gravity")
NAMESPACE_END(mitsuba)
