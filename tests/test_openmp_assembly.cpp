/*
	Thread-safety and determinism regression test for the element assembly.

	solver.cpp drives the element routine from an `omp parallel for` over the mesh,
	so evalWound must be reentrant: every write has to land in caller-owned storage,
	and the local Newton solve for the internal variables must not touch shared
	state. If that ever stops holding, the residual and tangent quietly become
	thread-count dependent and the Newton iteration loses quadratic convergence in
	a way that is very hard to trace back.

	The test evaluates the same mesh serially and then across every available
	thread count, and requires bit-for-bit identical residuals and tangents.
	Element evaluations are independent and each one reduces in a fixed order, so
	exact equality is the correct expectation here, not an approximate one.
*/

#include "wound.h"
#include "element_functions.h"
#include "myMeshGenerator.h"
#include "test_harness.h"

#include <vector>

namespace {

const int IP_SIZE = 8;      // 2x2x2 Gauss on the linear hexahedron
const int ELEM_SIZE = 8;
const int N_COORD = 3;

struct ElementResult {
	VectorXd Re_x, Re_rho, Re_c;
	MatrixXd Ke_x_x, Ke_rho_rho, Ke_c_c, Ke_x_rho;
};

std::vector<double> makeGlobalParameters()
{
	const double rho_phys = 1000 * 55.05126;
	const double c_max = 1.0e-4;

	double k0 = 0.00667792, kf = 0.015, k2 = 0.048;
	double t_rho = 1.28571e-5 / 55.05126;
	double t_rho_c = (1.28571e-5 * 3.28571) / 55.05126;
	double K_t = 0.2, K_t_c = c_max / 10.;
	double D_rhorho = 0.0833, D_rhoc = 0., D_cc = 0.01208;
	double p_rho = 0.04958333 / 55.05126;
	double p_rho_c = 0.015314, p_rho_theta = p_rho / 2.;
	double K_rho_c = c_max / 10., K_rho_rho = 10000 * 55.05126;
	double d_rho = p_rho * (1 - rho_phys / K_rho_rho);
	double vartheta_e = 2., gamma_theta = 5.;
	double p_c_rho = 90.0e-16 / rho_phys * 10;
	double p_c_thetaE = 300.0e-16 / rho_phys * 10;
	double K_c_c = 1., d_c = 0.01 / 2.;
	double bx = 0., by = 0., bz = 0.;

	return {k0, kf, k2, t_rho, t_rho_c, K_t, K_t_c, D_rhorho, D_rhoc, D_cc,
	        p_rho, p_rho_c, p_rho_theta, K_rho_c, K_rho_rho, d_rho, vartheta_e,
	        gamma_theta, p_c_rho, p_c_thetaE, K_c_c, d_c, bx, by, bz};
}

std::vector<double> makeLocalParameters()
{
	const double rho_phys = 1000 * 55.05126;

	double p_phi = 1.4e-8, p_phi_c = 7e-8, p_phi_theta = p_phi;
	double K_phi_c = 0.0001, d_phi = 3.7413e-4;
	double d_phi_rho_c = 0.5 * 0.000970 / rho_phys / 1.0e-4 / 10;
	double K_phi_rho = rho_phys * p_phi / d_phi - 1;
	double tau_omega = 10. / (K_phi_rho + 1);
	double tau_kappa = 1. / (K_phi_rho + 1);
	double gamma_kappa = 5.;
	double tau_lamdaP_a = 0.05, tau_lamdaP_s = 0.05, tau_lamdaP_n = 0.05;
	double vartheta_e = 2., gamma_theta = 5.;
	double tol_local = 1e-8, time_step_ratio = 100, max_iter = 100;

	return {p_phi, p_phi_c, p_phi_theta, K_phi_c, K_phi_rho, d_phi, d_phi_rho_c,
	        tau_omega, tau_kappa, gamma_kappa, tau_lamdaP_a, tau_lamdaP_s,
	        tau_lamdaP_n, vartheta_e, gamma_theta, tol_local, time_step_ratio, max_iter};
}

// Evaluate one element from a healthy-tissue state with a small imposed stretch,
// mirroring exactly how sparseWoundSolver sets the call up.
ElementResult evaluateElement(const HexMesh &mesh, int ei,
                              const std::vector<double> &global_parameters,
                              const std::vector<double> &local_parameters)
{
	const double rho_phys = 1000 * 55.05126;
	const double dt = 0.01, time = 0., time_final = 1.;

	std::vector<Vector3d> node_X_ni, node_x_ni;
	std::vector<double> node_rho_0_ni, node_c_0_ni, node_rho_ni, node_c_ni;

	for (int ni = 0; ni < ELEM_SIZE; ni++) {
		const Vector3d &X = mesh.nodes[mesh.elements[ei][ni]];
		node_X_ni.push_back(X);
		// A mild, spatially varying deformation so the tangent is not evaluated
		// at the trivial reference state.
		Vector3d x = X;
		x(0) *= 1.02;
		x(1) += 0.01 * X(0);
		node_x_ni.push_back(x);

		node_rho_0_ni.push_back(rho_phys);
		node_c_0_ni.push_back(0.);
		node_rho_ni.push_back(rho_phys);
		node_c_ni.push_back(1.0e-6);
	}

	std::vector<double> ip_phif_0(IP_SIZE, 1.0), ip_phif(IP_SIZE, 1.0);
	std::vector<double> ip_kappa_0(IP_SIZE, 1. / 3.), ip_kappa(IP_SIZE, 1. / 3.);
	std::vector<Vector3d> ip_a0_0(IP_SIZE, Vector3d(1., 0., 0.)), ip_a0(IP_SIZE, Vector3d(1., 0., 0.));
	std::vector<Vector3d> ip_s0_0(IP_SIZE, Vector3d(0., 1., 0.)), ip_s0(IP_SIZE, Vector3d(0., 1., 0.));
	std::vector<Vector3d> ip_n0_0(IP_SIZE, Vector3d(0., 0., 1.)), ip_n0(IP_SIZE, Vector3d(0., 0., 1.));
	std::vector<Vector3d> ip_lamdaP_0(IP_SIZE, Vector3d(1., 1., 1.)), ip_lamdaP(IP_SIZE, Vector3d(1., 1., 1.));
	std::vector<Vector3d> ip_lamdaE(IP_SIZE, Vector3d(1., 1., 1.));
	std::vector<Matrix3d> ip_strain(IP_SIZE, Matrix3d::Identity());
	std::vector<Matrix3d> ip_stress(IP_SIZE, Matrix3d::Zero());

	// The solver passes these in empty and lets the element fill them.
	std::vector<Vector3d> ip_dphifdu;
	std::vector<double> ip_dphifdrho, ip_dphifdc;

	std::vector<Matrix3d> ip_Jac = evalJacobian(node_X_ni);

	ElementResult r;
	r.Re_x = VectorXd::Zero(N_COORD * ELEM_SIZE);
	r.Re_rho = VectorXd::Zero(ELEM_SIZE);
	r.Re_c = VectorXd::Zero(ELEM_SIZE);
	r.Ke_x_x = MatrixXd::Zero(N_COORD * ELEM_SIZE, N_COORD * ELEM_SIZE);
	r.Ke_rho_rho = MatrixXd::Zero(ELEM_SIZE, ELEM_SIZE);
	r.Ke_c_c = MatrixXd::Zero(ELEM_SIZE, ELEM_SIZE);
	r.Ke_x_rho = MatrixXd::Zero(N_COORD * ELEM_SIZE, ELEM_SIZE);

	MatrixXd Ke_x_c = MatrixXd::Zero(N_COORD * ELEM_SIZE, ELEM_SIZE);
	MatrixXd Ke_rho_x = MatrixXd::Zero(ELEM_SIZE, N_COORD * ELEM_SIZE);
	MatrixXd Ke_rho_c = MatrixXd::Zero(ELEM_SIZE, ELEM_SIZE);
	MatrixXd Ke_c_x = MatrixXd::Zero(ELEM_SIZE, N_COORD * ELEM_SIZE);
	MatrixXd Ke_c_rho = MatrixXd::Zero(ELEM_SIZE, ELEM_SIZE);

	evalWound(dt, time, time_final,
	          ip_Jac, global_parameters, local_parameters,
	          ip_strain, ip_stress, node_rho_0_ni, node_c_0_ni,
	          ip_phif_0, ip_a0_0, ip_s0_0, ip_n0_0, ip_kappa_0, ip_lamdaP_0,
	          node_rho_ni, node_c_ni,
	          ip_phif, ip_a0, ip_s0, ip_n0, ip_kappa, ip_lamdaP,
	          ip_lamdaE,
	          node_x_ni, node_X_ni,
	          ip_dphifdu, ip_dphifdrho, ip_dphifdc,
	          r.Re_x, r.Ke_x_x, r.Ke_x_rho, Ke_x_c,
	          r.Re_rho, Ke_rho_x, r.Ke_rho_rho, Ke_rho_c,
	          r.Re_c, Ke_c_x, Ke_c_rho, r.Ke_c_c);

	return r;
}

// Exact equality: same code path, same reduction order, independent elements.
bool identical(const ElementResult &a, const ElementResult &b)
{
	return a.Re_x == b.Re_x && a.Re_rho == b.Re_rho && a.Re_c == b.Re_c &&
	       a.Ke_x_x == b.Ke_x_x && a.Ke_rho_rho == b.Ke_rho_rho &&
	       a.Ke_c_c == b.Ke_c_c && a.Ke_x_rho == b.Ke_x_rho;
}

} // namespace

int main()
{
	std::cout << "=== OpenMP assembly determinism ===\n";

#ifdef _OPENMP
	std::cout << "  built with OpenMP, max threads = " << omp_get_max_threads() << "\n";
#else
	std::cout << "  built without OpenMP: checking the serial path is reproducible\n";
#endif

	const std::vector<double> dims = {0., 2., 0., 2., 0., 1.};
	const std::vector<int> res = {4, 4, 3};
	HexMesh mesh = myHexMesh(dims, res);

	std::vector<double> global_parameters = makeGlobalParameters();
	std::vector<double> local_parameters = makeLocalParameters();

	// Reference: strictly serial evaluation.
	std::vector<ElementResult> reference(mesh.n_elements);
	for (int ei = 0; ei < mesh.n_elements; ei++) {
		reference[ei] = evaluateElement(mesh, ei, global_parameters, local_parameters);
	}

	// The residual must be non-trivial, otherwise the comparison proves nothing.
	double residual_magnitude = 0.;
	for (const ElementResult &r : reference) {
		residual_magnitude += r.Re_x.norm() + r.Re_rho.norm() + r.Re_c.norm();
	}
	std::cout << "  assembled " << mesh.n_elements
	          << " elements, total residual magnitude = " << residual_magnitude << "\n";
	CHECK(residual_magnitude > 0.);
	CHECK(std::isfinite(residual_magnitude));

	// Every tangent block must be finite; a NaN here means the local solve diverged.
	for (const ElementResult &r : reference) {
		CHECK(r.Ke_x_x.allFinite());
		CHECK(r.Ke_rho_rho.allFinite());
		CHECK(r.Ke_c_c.allFinite());
		CHECK(r.Re_x.allFinite());
	}

#ifdef _OPENMP
	const int max_threads = omp_get_max_threads();
	for (int n_threads = 1; n_threads <= max_threads; n_threads++) {
		omp_set_num_threads(n_threads);

		std::vector<ElementResult> parallel_result(mesh.n_elements);
#pragma omp parallel for
		for (int ei = 0; ei < mesh.n_elements; ei++) {
			parallel_result[ei] = evaluateElement(mesh, ei, global_parameters, local_parameters);
		}

		int n_mismatched = 0;
		for (int ei = 0; ei < mesh.n_elements; ei++) {
			if (!identical(reference[ei], parallel_result[ei])) { n_mismatched += 1; }
		}
		std::cout << "  " << n_threads << " thread(s): " << (mesh.n_elements - n_mismatched)
		          << "/" << mesh.n_elements << " elements bit-identical to serial\n";
		CHECK(n_mismatched == 0);
	}
#else
	std::vector<ElementResult> repeat(mesh.n_elements);
	for (int ei = 0; ei < mesh.n_elements; ei++) {
		repeat[ei] = evaluateElement(mesh, ei, global_parameters, local_parameters);
	}
	for (int ei = 0; ei < mesh.n_elements; ei++) {
		CHECK(identical(reference[ei], repeat[ei]));
	}
#endif

	return testing::summary("openmp_assembly");
}
