/*
	Unit tests for the isoparametric machinery: basis functions, quadrature
	rules, and the reference-configuration Jacobians.

	These are the properties every FEM discretization has to satisfy exactly,
	independent of the physics sitting on top of them.
*/

#include "element_functions.h"
#include "test_harness.h"

#include <vector>

namespace {

// A handful of deterministic sample points inside the parent domains.
const std::vector<Vector3d> hex_points = {
	Vector3d( 0.0,  0.0,  0.0),
	Vector3d( 0.3, -0.7,  0.5),
	Vector3d(-0.9,  0.2, -0.4),
	Vector3d( 0.6,  0.6,  0.6),
	Vector3d(-1.0,  1.0, -1.0)};

const std::vector<Vector3d> tet_points = {
	Vector3d(0.25, 0.25, 0.25),
	Vector3d(0.10, 0.20, 0.30),
	Vector3d(0.50, 0.10, 0.10),
	Vector3d(0.00, 0.00, 0.00)};

double sum(const std::vector<double> &v)
{
	double s = 0.;
	for (double vi : v) { s += vi; }
	return s;
}

// Sum_i N_i = 1 everywhere, and the derivatives of that identity sum to zero.
// Failure here means the element cannot even represent a rigid translation.
void checkPartitionOfUnity(const std::string &family,
                           std::vector<double> (*R)(double, double, double),
                           std::vector<double> (*Rxi)(double, double, double),
                           std::vector<double> (*Reta)(double, double, double),
                           std::vector<double> (*Rzeta)(double, double, double),
                           const std::vector<Vector3d> &points,
                           std::size_t expected_nodes)
{
	std::cout << "  [" << family << "] partition of unity\n";
	for (const Vector3d &p : points) {
		std::vector<double> N = R(p(0), p(1), p(2));
		CHECK(N.size() == expected_nodes);
		CHECK_CLOSE(sum(N), 1.0, 1e-12);
		CHECK_CLOSE(sum(Rxi(p(0), p(1), p(2))), 0.0, 1e-12);
		CHECK_CLOSE(sum(Reta(p(0), p(1), p(2))), 0.0, 1e-12);
		CHECK_CLOSE(sum(Rzeta(p(0), p(1), p(2))), 0.0, 1e-12);
	}
}

// N_i(xi_j) = delta_ij at the element nodes, which is what makes the
// coefficients of the expansion nodal values.
void checkKroneckerDeltaHex8()
{
	std::cout << "  [hex8] nodal interpolation property\n";
	const std::vector<Vector3d> node_Xi = {
		Vector3d(-1., -1., -1.), Vector3d(+1., -1., -1.), Vector3d(+1., +1., -1.), Vector3d(-1., +1., -1.),
		Vector3d(-1., -1., +1.), Vector3d(+1., -1., +1.), Vector3d(+1., +1., +1.), Vector3d(-1., +1., +1.)};

	for (std::size_t j = 0; j < node_Xi.size(); j++) {
		std::vector<double> N = evalShapeFunctionsR(node_Xi[j](0), node_Xi[j](1), node_Xi[j](2));
		for (std::size_t i = 0; i < N.size(); i++) {
			CHECK_CLOSE(N[i], (i == j ? 1.0 : 0.0), 1e-12);
		}
	}
}

// The quadrature weights have to integrate f = 1 over the parent domain:
// 8 for the reference cube, 1/6 for the reference tetrahedron.
void checkQuadratureWeights()
{
	std::cout << "  quadrature weights integrate unity over the parent domain\n";

	double w_hex_lin = 0.;
	for (const Vector4d &ip : LineQuadriIP()) { w_hex_lin += ip(3); }
	CHECK_CLOSE(w_hex_lin, 8.0, 1e-12);

	double w_hex_quad = 0.;
	for (const Vector4d &ip : LineQuadriIPQuadratic()) { w_hex_quad += ip(3); }
	CHECK_CLOSE(w_hex_quad, 8.0, 1e-12);

	double w_tet_lin = 0.;
	for (const Vector4d &ip : LineQuadriIPTet()) { w_tet_lin += ip(3); }
	CHECK_CLOSE(w_tet_lin, 1. / 6., 1e-12);

	double w_tet_quad = 0.;
	for (const Vector4d &ip : LineQuadriIPTetQuadratic()) { w_tet_quad += ip(3); }
	CHECK_CLOSE(w_tet_quad, 1. / 6., 1e-12);
}

// 2x2x2 Gauss is exact through cubic order in each direction. Integrating
// monomials against the hex8 basis pins down both the points and the weights.
void checkGaussExactness()
{
	std::cout << "  [hex8] 2x2x2 Gauss integrates xi^2 eta^2 zeta^2 exactly\n";
	std::vector<Vector4d> IP = LineQuadriIP();

	double integral_quadratic = 0.;
	double integral_linear = 0.;
	for (const Vector4d &ip : IP) {
		double xi = ip(0), eta = ip(1), zeta = ip(2), w = ip(3);
		integral_quadratic += w * xi * xi * eta * eta * zeta * zeta;
		integral_linear += w * xi * eta * zeta;
	}
	// int_{-1}^{1}x^2 dx = 2/3, cubed.
	CHECK_CLOSE(integral_quadratic, (2. / 3.) * (2. / 3.) * (2. / 3.), 1e-12);
	// Odd integrand over a symmetric domain.
	CHECK_CLOSE(integral_linear, 0.0, 1e-12);
}

// evalJacobian returns the inverse transpose of dX/dxi at each integration
// point. For a cube of side L the map is X = (L/2)xi, so det(J^-T) = (2/L)^3
// and sum_ip w_ip * det(J) recovers the physical element volume.
void checkJacobianOnCube()
{
	std::cout << "  [hex8] Jacobian and volume recovery on a cube\n";
	for (double L : {1.0, 2.0, 0.25}) {
		std::vector<Vector3d> node_X = {
			Vector3d(0., 0., 0.), Vector3d(L, 0., 0.), Vector3d(L, L, 0.), Vector3d(0., L, 0.),
			Vector3d(0., 0., L),  Vector3d(L, 0., L),  Vector3d(L, L, L),  Vector3d(0., L, L)};

		std::vector<Matrix3d> ip_Jac = evalJacobian(node_X);
		CHECK(ip_Jac.size() == 8u);

		std::vector<Vector4d> IP = LineQuadriIP();
		double volume = 0.;
		for (std::size_t ip = 0; ip < ip_Jac.size(); ip++) {
			double det_JiT = ip_Jac[ip].determinant();
			// Orientation must be positive: a negative determinant is an inverted element.
			CHECK(det_JiT > 0.);
			CHECK_CLOSE(det_JiT, 8. / (L * L * L), 1e-10);
			volume += IP[ip](3) / det_JiT;
		}
		CHECK_CLOSE(volume, L * L * L, 1e-10);
	}
}

// A translated and rotated cube is still a cube: the volume is unchanged.
void checkJacobianFrameInvariance()
{
	std::cout << "  [hex8] element volume is invariant to rigid body motion\n";
	const double L = 1.3;
	std::vector<Vector3d> node_X = {
		Vector3d(0., 0., 0.), Vector3d(L, 0., 0.), Vector3d(L, L, 0.), Vector3d(0., L, 0.),
		Vector3d(0., 0., L),  Vector3d(L, 0., L),  Vector3d(L, L, L),  Vector3d(0., L, L)};

	const double th = 0.7;
	Matrix3d Q;
	Q << std::cos(th), -std::sin(th), 0.,
	     std::sin(th),  std::cos(th), 0.,
	     0.,            0.,           1.;
	Vector3d t(3.1, -2.2, 0.4);

	std::vector<Vector3d> node_X_moved;
	for (const Vector3d &X : node_X) { node_X_moved.push_back(Q * X + t); }

	std::vector<Matrix3d> Jac_ref = evalJacobian(node_X);
	std::vector<Matrix3d> Jac_moved = evalJacobian(node_X_moved);

	std::vector<Vector4d> IP = LineQuadriIP();
	double vol_ref = 0., vol_moved = 0.;
	for (std::size_t ip = 0; ip < IP.size(); ip++) {
		vol_ref += IP[ip](3) / Jac_ref[ip].determinant();
		vol_moved += IP[ip](3) / Jac_moved[ip].determinant();
	}
	CHECK_CLOSE(vol_moved, vol_ref, 1e-10);
	CHECK_CLOSE(vol_ref, L * L * L, 1e-10);
}

} // namespace

int main()
{
	std::cout << "=== element_functions ===\n";

	checkPartitionOfUnity("hex8", evalShapeFunctionsR, evalShapeFunctionsRxi,
	                      evalShapeFunctionsReta, evalShapeFunctionsRzeta, hex_points, 8);
	checkPartitionOfUnity("hex20", evalShapeFunctionsQuadraticR, evalShapeFunctionsQuadraticRxi,
	                      evalShapeFunctionsQuadraticReta, evalShapeFunctionsQuadraticRzeta, hex_points, 20);
	checkPartitionOfUnity("hex27", evalShapeFunctionsQuadraticLagrangeR, evalShapeFunctionsQuadraticLagrangeRxi,
	                      evalShapeFunctionsQuadraticLagrangeReta, evalShapeFunctionsQuadraticLagrangeRzeta,
	                      hex_points, 27);
	checkPartitionOfUnity("tet4", evalShapeFunctionsTetR, evalShapeFunctionsTetRxi,
	                      evalShapeFunctionsTetReta, evalShapeFunctionsTetRzeta, tet_points, 4);
	checkPartitionOfUnity("tet10", evalShapeFunctionsTetQuadraticR, evalShapeFunctionsTetQuadraticRxi,
	                      evalShapeFunctionsTetQuadraticReta, evalShapeFunctionsTetQuadraticRzeta, tet_points, 10);

	checkKroneckerDeltaHex8();
	checkQuadratureWeights();
	checkGaussExactness();
	checkJacobianOnCube();
	checkJacobianFrameInvariance();

	return testing::summary("element_functions");
}
