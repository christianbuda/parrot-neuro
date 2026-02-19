#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <variant>
#include <set>

// TBB (Threading)
#include <tbb/global_control.h>
#include <tbb/info.h>

// CGAL Core
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/Mesh_3/generate_label_weights.h>
#include <CGAL/IO/File_medit.h>
#include <CGAL/tags.h> 

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Mesh_domain = CGAL::Labeled_mesh_domain_3<K>;

// Parallel Tag
using Tr = CGAL::Mesh_triangulation_3<
    Mesh_domain, 
    CGAL::Default, 
    CGAL::Parallel_if_available_tag
>::type;

using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;
namespace params = CGAL::parameters;

// --- ADVANCED SIZING FIELD ---
// Decouples Surface Resolution from Volume Resolution
class Sizing_Field {
public:
    typedef K::Point_3 Point_3;
    typedef Mesh_domain::Index Index;

    double def_surf;
    double def_vol;

    // We store pairs: lookup[label] = {surface_size, volume_size}
    struct SizePair {
        double surf;
        double vol;
    };
    std::vector<SizePair> lookup_table; 

    Sizing_Field(double default_surface_size, double default_volume_size) 
        : def_surf(default_surface_size), def_vol(default_volume_size) {
        lookup_table.resize(256, {def_surf, def_vol});
    }

    void add_label_specs(int label, double surf_size, double vol_size) {
        if (label >= lookup_table.size()) {
            lookup_table.resize(label + 1, {def_surf, def_vol});
        }
        lookup_table[label] = {surf_size, vol_size};
    }

    // CGAL calls this operator to ask for size at a specific point/index
    inline double operator()(const Point_3&, const int, const Index& index) const {

        // 1. VOLUME CASE (Inside a tissue)
        // The index holds an int corresponding to the subdomain label
        if (const int* ptr = std::get_if<int>(&index)) {
            int label = *ptr;
            if (label >= 0 && label < lookup_table.size()) {
                return lookup_table[label].vol;
            }
            return def_vol;
        } 
        
        // 2. SURFACE CASE (On the boundary)
        // The index holds a pair<int,int> corresponding to the two tissues touching
        else if (const std::pair<int, int>* ptr = std::get_if<std::pair<int, int>>(&index)) {
            int label_a = ptr->first;
            int label_b = ptr->second;

            double size_a = (label_a >= 0 && label_a < lookup_table.size()) ? lookup_table[label_a].surf : def_surf;
            double size_b = (label_b >= 0 && label_b < lookup_table.size()) ? lookup_table[label_b].surf : def_surf;

            // On a boundary, we must satisfy the STRICTER (smaller) constraint
                return (size_a < size_b) ? size_a : size_b;
            }
        
        return def_surf; // Fallback
    }
};

int main(int argc, char* argv[]) {
    // UPDATED USAGE:
    // 1. Cores
    // 2. Input (.inr)
    // 3. Output (.mesh)
    // 4. Facet Angle (deg)
    // 5. Facet Distance (mm) - Approximation error
    // 6. Facet Size Default (mm) - Surface max size
    // 7. Cell Size Default (mm) - Volume max size
    // 8. Cell Radius/Edge Ratio - Shape quality (default 3)
    // 9. Smoothing factor - Radius of smoothing prior to meshing
    // 10. Optimization Time (seconds) - Max time per optimization step
    // 11+ Label Overrides: "Label:SurfSize:VolSize"

    if (argc < 11) {
        std::cerr << "Usage: " << argv[0] << " <CORES> <IN> <OUT> <ANGLE> <DIST> <DEF_SURF> <DEF_VOL> <RATIO> <SMOOTH> <OPT_TIME> [L:S:V]..." << std::endl;
        return 1;
    }

    // --- 1. SETUP & PARSING ---
    int max_cores = std::stoi(argv[1]);
    std::unique_ptr<tbb::global_control> tbb_control;
    if (max_cores > 0) {
        tbb_control = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, max_cores);
        std::cout << "Threads limited to: " << max_cores << std::endl;
    }

    std::string input_file = argv[2];
    std::string output_file = argv[3];
    
    double angle = std::stod(argv[4]);
    double dist = std::stod(argv[5]);
    double def_surf = std::stod(argv[6]);
    double def_vol = std::stod(argv[7]);
    double ratio = std::stod(argv[8]);
    double smoothing_factor = std::stod(argv[9]);
    double opt_time = std::stod(argv[10]);

    std::cout << "--- PARAMETERS ---" << std::endl;
    std::cout << "Angle: " << angle << " | Dist: " << dist << std::endl;
    std::cout << "Default Surface: " << def_surf << " | Default Volume: " << def_vol << std::endl;
    std::cout << "Optimization Time Limit: " << opt_time << "s per step" << std::endl;

    Sizing_Field size_func(def_surf, def_vol);
    
    // Parse overrides (Format: Label:Surf:Vol)
    for (int i = 11; i < argc; ++i) {
        std::string arg = argv[i];
        size_t p1 = arg.find(':');
        size_t p2 = arg.find(':', p1 + 1);

        if (p1 != std::string::npos && p2 != std::string::npos) {
            int label = std::stoi(arg.substr(0, p1));
            double s_surf = std::stod(arg.substr(p1 + 1, p2 - (p1 + 1)));
            double s_vol = std::stod(arg.substr(p2 + 1));
            
            size_func.add_label_specs(label, s_surf, s_vol);
            std::cout << " -> Override Label " << label << ": Surface=" << s_surf << ", Volume=" << s_vol << std::endl;
        }
    }

    // --- 2. LOAD & PREPARE DOMAIN ---
    std::cout << "Reading image..." << std::endl;
    CGAL::Image_3 image;
    if(!image.read(input_file.c_str())) return 1;

    // Generate weights for smooth surfaces
    const float min_voxel = (std::min)(image.vx(), (std::min)(image.vy(), image.vz()));
    const float sigma = min_voxel * (float)smoothing_factor;
    
    std::cout << "Smoothing Sigma: " << sigma << " (Factor: " << smoothing_factor << "x)" << std::endl;

    CGAL::Image_3 img_weights = CGAL::Mesh_3::generate_label_weights(image, sigma);

    Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(
        image,
        params::weights(img_weights).relative_error_bound(1e-6)
    );

    // --- 3. CRITERIA ---
    Mesh_criteria criteria(
        params::facet_angle(angle).
        facet_size(size_func).       // Uses Sizing_Field for Surfaces
        facet_distance(dist).
        cell_radius_edge_ratio(ratio).
        cell_size(size_func)        // Uses Sizing_Field for Volumes
    );

    // --- 4. MESHING (Refinement + Topology) ---
    std::cout << "Starting Initial Meshing (Delaunay Refinement)..." << std::endl;
    
    // We disable implicit optimization here to run it manually later
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(
        domain, 
        criteria, 
        params::no_odt().no_lloyd().no_perturb().no_exude()
    );

    // --- 5. EXPLICIT OPTIMIZATION PIPELINE ---
    // This gives us control over what runs and for how long.

    std::cout << "1. Optimization: ODT (Global Smoothing)..." << std::endl;
    // ODT moves vertices to optimal positions
    CGAL::odt_optimize_mesh_3(c3t3, domain, params::time_limit(opt_time)); 

    std::cout << "2. Optimization: Perturb (Sliver Removal)..." << std::endl;
    // Perturb jiggles vertices to kill slivers (flat tets)
    CGAL::perturb_mesh_3(c3t3, domain, params::time_limit(opt_time));

    std::cout << "3. Optimization: Exude (Vertex Weighting)..." << std::endl;
    // Exude adjusts vertex weights to maximize min-angle
    CGAL::exude_mesh_3(c3t3, params::time_limit(opt_time));


    // --- 6. OUTPUT ---
    std::cout << "Writing output to " << output_file << "..." << std::endl;
    std::ofstream medit_file(output_file);
    CGAL::IO::write_MEDIT(medit_file, c3t3);

    return 0;
}
