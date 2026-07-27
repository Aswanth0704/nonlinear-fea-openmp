/*
	Integration tests for the structured mesh generator.

	A mesh is only usable if the connectivity ordering matches the parent element,
	so these tests go through evalJacobian and require a positive determinant at
	every integration point of every element, then check that the quadrature over
	the whole mesh recovers the analytic domain volume.
*/

#include "myMeshGenerator.h"
#include "element_functions.h"
#include "test_harness.h"

#include <set>
#include <vector>

namespace {

// Node and element counts follow directly from the requested resolution.
void checkMeshCounts()
{
	std::cout << "  node and element counts match the requested resolution\n";

	const std::vector<double> dims = {0., 2., 0., 3., 0., 1.};
	const std::vector<int> res = {4, 5, 3};

	HexMesh mesh = myHexMesh(dims, res);

	CHECK(mesh.n_nodes == res[0] * res[1] * res[2]);
	CHECK(mesh.n_elements == (res[0] - 1) * (res[1] - 1) * (res[2] - 1));
	CHECK(mesh.nodes.size() == static_cast<std::size_t>(mesh.n_nodes));
	CHECK(mesh.elements.size() == static_cast<std::size_t>(mesh.n_elements));
	CHECK(mesh.boundary_flag.size() == static_cast<std::size_t>(mesh.n_nodes));
}

// Every connectivity entry must index a real node, and no element may
// reference the same node twice.
void checkConnectivityIsValid()
{
	std::cout << "  connectivity indices are in range and non-degenerate\n";

	const std::vector<double> dims = {-1., 1., -1., 1., 0., 2.};
	const std::vector<int> res = {3, 3, 4};

	HexMesh mesh = myHexMesh(dims, res);

	for (const std::vector<int> &elem : mesh.elements) {
		CHECK(elem.size() == 8u);
		std::set<int> unique_nodes;
		for (int node : elem) {
			CHECK(node >= 0);
			CHECK(node < mesh.n_nodes);
			unique_nodes.insert(node);
		}
		CHECK(unique_nodes.size() == 8u);
	}
}

// Node coordinates must span exactly the requested bounding box.
void checkNodeExtents()
{
	std::cout << "  generated nodes span the requested domain\n";

	const std::vector<double> dims = {-0.5, 2.5, 1., 4., -2., 0.};
	const std::vector<int> res = {5, 4, 3};

	HexMesh mesh = myHexMesh(dims, res);

	Vector3d lo = mesh.nodes[0], hi = mesh.nodes[0];
	for (const Vector3d &X : mesh.nodes) {
		lo = lo.cwiseMin(X);
		hi = hi.cwiseMax(X);
	}

	CHECK_CLOSE(lo(0), dims[0], 1e-12);  CHECK_CLOSE(hi(0), dims[1], 1e-12);
	CHECK_CLOSE(lo(1), dims[2], 1e-12);  CHECK_CLOSE(hi(1), dims[3], 1e-12);
	CHECK_CLOSE(lo(2), dims[4], 1e-12);  CHECK_CLOSE(hi(2), dims[5], 1e-12);
}

// The real test of connectivity ordering: no inverted elements anywhere, and
// the assembled quadrature reproduces the analytic volume of the box.
void checkNoInvertedElementsAndVolume()
{
	std::cout << "  no inverted elements, and quadrature recovers the domain volume\n";

	const std::vector<double> dims = {0., 2., 0., 3., 0., 1.5};
	const std::vector<int> res = {4, 5, 3};

	HexMesh mesh = myHexMesh(dims, res);
	std::vector<Vector4d> IP = LineQuadriIP();

	double total_volume = 0.;
	for (const std::vector<int> &elem : mesh.elements) {
		std::vector<Vector3d> node_X;
		for (int node : elem) { node_X.push_back(mesh.nodes[node]); }

		std::vector<Matrix3d> ip_Jac = evalJacobian(node_X);
		for (std::size_t ip = 0; ip < ip_Jac.size(); ip++) {
			double det_JiT = ip_Jac[ip].determinant();
			CHECK(det_JiT > 0.);              // negative means the element is inverted
			total_volume += IP[ip](3) / det_JiT;
		}
	}

	const double analytic = (dims[1] - dims[0]) * (dims[3] - dims[2]) * (dims[5] - dims[4]);
	CHECK_CLOSE(total_volume, analytic, 1e-10);
}

// Boundary flags: interior nodes carry 0, and every face of the box is tagged.
void checkBoundaryFlags()
{
	std::cout << "  boundary flags tag all six faces and leave the interior clear\n";

	const std::vector<double> dims = {0., 1., 0., 1., 0., 1.};
	const std::vector<int> res = {4, 4, 4};

	HexMesh mesh = myHexMesh(dims, res);

	std::set<int> flags_seen;
	int n_interior = 0;
	for (int i = 0; i < mesh.n_nodes; i++) {
		int flag = mesh.boundary_flag[i];
		flags_seen.insert(flag);

		const Vector3d &X = mesh.nodes[i];
		bool on_boundary = (std::fabs(X(0) - dims[0]) < 1e-12 || std::fabs(X(0) - dims[1]) < 1e-12 ||
		                    std::fabs(X(1) - dims[2]) < 1e-12 || std::fabs(X(1) - dims[3]) < 1e-12 ||
		                    std::fabs(X(2) - dims[4]) < 1e-12 || std::fabs(X(2) - dims[5]) < 1e-12);

		// A node is tagged if and only if it actually sits on the boundary.
		CHECK((flag != 0) == on_boundary);
		if (flag == 0) { n_interior += 1; }
	}

	// All six face tags must appear, plus 0 for the interior.
	for (int expected : {0, 1, 2, 3, 4, 5, 6}) {
		CHECK(flags_seen.count(expected) == 1u);
	}
	CHECK(n_interior == (res[0] - 2) * (res[1] - 2) * (res[2] - 2));
}

} // namespace

int main()
{
	std::cout << "=== mesh generator ===\n";

	checkMeshCounts();
	checkConnectivityIsValid();
	checkNodeExtents();
	checkNoInvertedElementsAndVolume();
	checkBoundaryFlags();

	return testing::summary("mesh_generator");
}
