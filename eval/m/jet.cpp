
#define CGAL_EIGEN3_ENABLED

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Monge_via_jet_fitting.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <fstream>
#include <algorithm>
#include <vector>
typedef double                   DFT;
typedef CGAL::Simple_cartesian<DFT>     Data_Kernel;
typedef Data_Kernel::Point_3     DPoint;
typedef CGAL::Monge_via_jet_fitting<Data_Kernel> My_Monge_via_jet_fitting;
typedef My_Monge_via_jet_fitting::Monge_form     My_Monge_form;


typedef CGAL::Search_traits_3<Data_Kernel> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;


// // DPoint op(Neighbor_search::iterator & it) {
// //   return it->first;
// }

int main(int argc, char *argv[])
{
  size_t d_fitting = 4;
  size_t d_monge = 4;
  size_t K = 16;
  const char* name_file_in = "data/in_points_d4.txt";
  const char* name_file_out = "data/out_points_d4.txt";
  //check command line
  if (argc<4)
    {
      std::cout << " Usage : " << argv[0]
                << " <inputPoints.xyz> <d_fitting> <d_monge> <K> <output.xyz>" << std::endl
                << "test with default arguments" << std::endl;
    }
  else {
    name_file_in = argv[1];
    d_fitting = std::atoi(argv[2]);
    d_monge = std::atoi(argv[3]);
    K = std::atoi(argv[4]);
    name_file_out = argv[5];
  }
  //open the input file
  std::ifstream inFile(name_file_in);
  if ( !inFile )
    {
      std::cerr << "cannot open file for input\n";
      exit(-1);
    }
  //initalize the in_points container
  double x, y, z;
  std::vector<DPoint> in_points;
  while (inFile >> x) {
    inFile >> y >> z;
    DPoint p(x,y,z);
    in_points.push_back(p);
  }
  inFile.close();

  std::vector<DPoint> out_points;

  Neighbor_search::Tree knn_tree(in_points.begin(), in_points.end());

  for (size_t i = 0; i < in_points.size(); i++) {
    DPoint query = in_points[i];
    Neighbor_search search(knn_tree, query, K);

    std::vector<DPoint> knn_points = {};
    // knn_points.resize(K);
    // std::transform(search.begin(), search.end(), knn_points.begin(), op);
    for (Neighbor_search::iterator it = search.begin(); it != search.end(); ++it) {
      knn_points.push_back(it->first);
    }

    My_Monge_form monge_form;
    My_Monge_via_jet_fitting monge_fit;
    monge_form = monge_fit(knn_points.begin(), knn_points.end(), d_fitting, d_monge);    
    DPoint fitted_pt = monge_form.origin();

    out_points.push_back(fitted_pt);
  }

  std::ofstream outFile(name_file_out);
  write_xyz_points(outFile, out_points);

  // // fct parameters
  // My_Monge_form monge_form;
  // My_Monge_via_jet_fitting monge_fit;
  // monge_form = monge_fit(in_points.begin(), in_points.end(), d_fitting, d_monge);
  // //OUTPUT on std::cout
  // CGAL::set_pretty_mode(std::cout);

  // std::cout << "vertex : " << in_points[0] << std::endl
  //           << "number of points used : " << in_points.size() << std::endl
  //           << monge_form;
  // std::cout  << "condition_number : " << monge_fit.condition_number() << std::endl
  //            << "pca_eigen_vals and associated pca_eigen_vecs :"  << std::endl;
  // for (int i=0; i<3; i++)
  //   std::cout << monge_fit.pca_basis(i).first << std::endl
  //             << monge_fit.pca_basis(i).second  << std::endl;
  return 0;
}
