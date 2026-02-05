#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <variant>

// TBB
#include <tbb/global_control.h>
#include <tbb/info.h> // To query default threads

// CGAL
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

// --- OPTIMIZED SIZING FIELD ---
// Uses a flat vector instead of a map for O(1) lookup speed.
class Sizing_Field {
public:
    typedef K::Point_3 Point_3;
    typedef Mesh_domain::Index Index;

    double default_size;
    // Lookup table: index = label ID, value = size
    std::vector<double> lookup_table; 

    Sizing_Field(double def) : default_size(def) {
        // Pre-allocate space for 256 labels (standard for INR/8-bit images)
        // If your labels go higher (e.g. 1000), increase this number.
        lookup_table.resize(256, default_size);
    }

    void add_label_size(int label, double size) {
        if (label >= lookup_table.size()) {
            lookup_table.resize(label + 1, default_size);
        }
        lookup_table[label] = size;
    }

    // Inlined for speed
    inline double operator()(const Point_3&, const int, const Index& index) const {
        int label_a = -1;
        int label_b = -1;

        // Fast Variant Access
        if (const int* ptr = std::get_if<int>(&index)) {
            // Volume case
            label_a = *ptr;
        } else if (const std::pair<int, int>* ptr = std::get_if<std::pair<int, int>>(&index)) {
            // Surface case
            label_a = ptr->first;
            label_b = ptr->second;
        }

        // Safety check
        if (label_a < 0 || label_a >= lookup_table.size()) return default_size;

        double size_a = lookup_table[label_a];

        if (label_b != -1) {
            // On a boundary, take the smaller (more detailed) size
            if (label_b >= 0 && label_b < lookup_table.size()) {
                double size_b = lookup_table[label_b];
                return (size_a < size_b) ? size_a : size_b;
            }
        }
        
        return size_a;
    }
};

int main(int argc, char* argv[]) {
    // Usage: <CORES> <in.inr> <out.mesh> <angle> <def_size> <dist> [label:size]...
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <CORES> <in.inr> <out.mesh> ... " << std::endl;
        return 1;
    }

    // 1. THREAD CONTROL & DIAGNOSTICS
    int max_cores = std::stoi(argv[1]);
    int system_cores = tbb::info::default_concurrency();
    
    std::cout << "--- SYSTEM INFO ---" << std::endl;
    std::cout << "Detected System Cores: " << system_cores << std::endl;

    std::unique_ptr<tbb::global_control> tbb_control;
    if (max_cores > 0) {
        std::cout << "Forcing limit: " << max_cores << " threads." << std::endl;
        tbb_control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, max_cores
        );
    } else {
        std::cout << "Allowed to use ALL available threads." << std::endl;
    }
    std::cout << "-------------------" << std::endl;

    // 2. Parse standard args
    std::string input_file = argv[2];
    std::string output_file = argv[3];
    double angle = std::stod(argv[4]);
    double default_size = std::stod(argv[5]);
    double dist = std::stod(argv[6]);

    Sizing_Field size_func(default_size);
    
    // Parse label overrides
    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        size_t colon_pos = arg.find(':');
        if (colon_pos != std::string::npos) {
            int label = std::stoi(arg.substr(0, colon_pos));
            double size = std::stod(arg.substr(colon_pos + 1));
            size_func.add_label_size(label, size);
            std::cout << " -> Label " << label << " resolution: " << size << "mm" << std::endl;
        }
    }

    std::cout << "Reading " << input_file << "..." << std::endl;
    CGAL::Image_3 image;
    if(!image.read(input_file.c_str())) return 1;

    // Use conservative sigma for thin layers
    const float sigma = (std::min)(image.vx(), (std::min)(image.vy(), image.vz()));
    
    CGAL::Image_3 img_weights = CGAL::Mesh_3::generate_label_weights(image, sigma);

    Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(
        image,
        params::weights(img_weights).relative_error_bound(1e-6)
    );

    Mesh_criteria criteria(
        params::facet_angle(angle).
        facet_size(size_func).       
        facet_distance(dist).
        cell_radius_edge_ratio(3).
        cell_size(size_func)         
    );

    std::cout << "Meshing (Refinement + Optimization)..." << std::endl;
    
    try {
    // PRODUCTION SETTINGS FOR FEM (DUNEuro)
    // 1. odt(): Optimizes mesh quality (makes tets equilateral).
    // 2. perturb() & exude(): ENABLED implicitly (we removed the no_ flags).
    // 3. manifold(): Ensures watertight topology (optional, but good for FEM).
    
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(
        domain, 
        criteria, 
        params::odt(),      // Enable ODT smoothing
        params::manifold()  // Force manifold topology
    );

    std::cout << "Done! Saving output..." << std::endl;
    std::ofstream medit_file(output_file);
    CGAL::IO::write_MEDIT(medit_file, c3t3);
    
    } catch (const std::bad_alloc&) {
        std::cerr << "FATAL ERROR: Out of Memory! The mesh is too complex for your RAM." << std::endl;
        std::cerr << "Try increasing facet_size or facet_distance." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: C++ Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "FATAL ERROR: Unknown crash occurred." << std::endl;
        return 1;
    }
    return 0;
}
