/*
	Regression tests for the finite-strain constitutive response.

	The model splits the deformation multiplicatively, FF = FFe * FFg, with the
	growth/contracture part FFg built from the stretches lamdaP along an evolving
	orthonormal fiber triad. These tests pin down the properties that split has to
	satisfy exactly, in floating point, for the Newton tangent to be consistent:

	  1. the undeformed reference state carries no stress,
	  2. a purely grown configuration (FFe = I) carries no stress either,
	  3. the second Piola-Kirchhoff stress is objective under rotation of the
	     reference frame,
	  4. the response is monotone in a uniaxial fiber stretch.

	Property 2 is the one that actually exercises the split: the neo-Hookean and
	volumetric contributions have to cancel after the pull-back through FFg.
*/

#include "wound.h"
#include "test_harness.h"

#include <limits>
#include <vector>

namespace {

// Parameter ordering follows global_parameters in src/results_circle_wound.cpp.
// Only entries 0..6 are read by evalSS; the rest are padding.
std::vector<double> makeGlobalParameters()
{
	const double rho_phys = 1000 * 55.05126;
	const double c_max = 1.0e-4;

	double k0 = 0.00667792;   // neo-Hookean ground substance [MPa]
	double kf = 0.015;        // collagen stiffness [MPa]
	double k2 = 0.048;        // exponential stiffening exponent [-]
	double t_rho = 1.28571e-5 / 55.05126;
	double t_rho_c = (1.28571e-5 * 3.28571) / 55.05126;
	double K_t = 0.2;
	double K_t_c = c_max / 10.;

	std::vector<double> p(25, 0.);
	p[0] = k0; p[1] = kf; p[2] = k2;
	p[3] = t_rho; p[4] = t_rho_c; p[5] = K_t; p[6] = K_t_c;
	(void)rho_phys;
	return p;
}

// Rotation about a normalized axis by angle th (Rodrigues).
Matrix3d rotation(Vector3d axis, double th)
{
	axis = axis / axis.norm();
	Matrix3d K;
	K <<        0., -axis(2),  axis(1),
	      axis(2),        0., -axis(0),
	     -axis(1),  axis(0),        0.;
	return Matrix3d::Identity() + std::sin(th) * K + (1. - std::cos(th)) * K * K;
}

double maxAbs(const Matrix3d &A)
{
	return A.cwiseAbs().maxCoeff();
}

// 1. Undeformed reference state: CC = I, no growth, no cells or chemical.
//    Passive and volumetric parts must cancel exactly and the active part vanishes.
void checkStressFreeReference()
{
	std::cout << "  undeformed reference state is stress free\n";
	std::vector<double> gp = makeGlobalParameters();

	Vector3d a0(1., 0., 0.), s0(0., 1., 0.), n0(0., 0., 1.);
	Matrix3d CC = Matrix3d::Identity();

	// The cancellation is independent of the dispersion parameter, so sweep it.
	for (double kappa : {0.0, 1. / 6., 1. / 3.}) {
		for (double phif : {0.01, 0.5, 1.0}) {
			Matrix3d SS_pas, SS_act, SS_vol;
			evalSS(gp, phif, a0, s0, n0, kappa, 1., 1., 1., CC, /*rho*/ 0., /*c*/ 0.,
			       SS_pas, SS_act, SS_vol);

			CHECK_CLOSE(maxAbs(SS_pas + SS_vol), 0.0, 1e-14);
			CHECK_CLOSE(maxAbs(SS_act), 0.0, 1e-14);
		}
	}
}

// 2. A configuration reached by pure growth, CC = FFg^T FFg, leaves FFe = I.
//    The elastic stress must vanish for any growth stretch and any fiber
//    orientation. This is the invariant that the multiplicative split exists to
//    guarantee, and it is where an inconsistent pull-back would show up.
void checkStressFreeGrownConfiguration()
{
	std::cout << "  purely grown configuration (FFe = I) is stress free\n";
	std::vector<double> gp = makeGlobalParameters();

	const std::vector<Vector3d> lamdaPs = {
		Vector3d(1.0, 1.0, 1.0),
		Vector3d(1.3, 0.9, 1.1),
		Vector3d(0.7, 1.5, 1.0),
		Vector3d(1.05, 1.05, 0.92)};

	const std::vector<Matrix3d> frames = {
		Matrix3d::Identity(),
		rotation(Vector3d(0., 0., 1.), 0.61),
		rotation(Vector3d(1., 1., 1.), 2.1),
		rotation(Vector3d(0.3, -0.8, 0.5), 1.27)};

	for (const Matrix3d &Q : frames) {
		Vector3d a0 = Q.col(0);
		Vector3d s0 = Q.col(1);
		Vector3d n0 = Q.col(2);

		// The triad has to stay orthonormal or the split is meaningless.
		CHECK_CLOSE(a0.dot(s0), 0.0, 1e-13);
		CHECK_CLOSE(a0.dot(n0), 0.0, 1e-13);
		CHECK_CLOSE(a0.norm(), 1.0, 1e-13);

		for (const Vector3d &lamdaP : lamdaPs) {
			Matrix3d FFg = lamdaP(0) * (a0 * a0.transpose())
			             + lamdaP(1) * (s0 * s0.transpose())
			             + lamdaP(2) * (n0 * n0.transpose());
			Matrix3d CC = FFg.transpose() * FFg;

			Matrix3d SS_pas, SS_act, SS_vol;
			evalSS(gp, /*phif*/ 1.0, a0, s0, n0, /*kappa*/ 1. / 3.,
			       lamdaP(0), lamdaP(1), lamdaP(2), CC, /*rho*/ 0., /*c*/ 0.,
			       SS_pas, SS_act, SS_vol);

			CHECK_CLOSE(maxAbs(SS_pas + SS_vol), 0.0, 1e-13);
		}
	}
}

// 3. Objectivity: rotating the reference frame must rotate the stress with it.
//    SS(R CC R^T, R a0, ...) = R SS(CC, a0, ...) R^T, including the active part.
void checkObjectivity()
{
	std::cout << "  second Piola-Kirchhoff stress is objective\n";
	std::vector<double> gp = makeGlobalParameters();

	Vector3d a0(1., 0., 0.), s0(0., 1., 0.), n0(0., 0., 1.);
	Vector3d lamdaP(1.15, 0.95, 1.02);
	const double kappa = 0.21, phif = 0.8, rho = 1200., c = 2.0e-5;

	// A genuinely deformed, non-symmetric-in-the-axes state.
	Matrix3d FF;
	FF << 1.18, 0.07, -0.03,
	     -0.05, 0.92,  0.11,
	      0.02, 0.06,  1.07;
	Matrix3d CC = FF.transpose() * FF;

	Matrix3d SS_pas, SS_act, SS_vol;
	evalSS(gp, phif, a0, s0, n0, kappa, lamdaP(0), lamdaP(1), lamdaP(2), CC, rho, c,
	       SS_pas, SS_act, SS_vol);

	for (double th : {0.4, 1.9, -2.6}) {
		Matrix3d R = rotation(Vector3d(0.2, 0.9, -0.4), th);

		Matrix3d SS_pas_r, SS_act_r, SS_vol_r;
		evalSS(gp, phif, R * a0, R * s0, R * n0, kappa, lamdaP(0), lamdaP(1), lamdaP(2),
		       R * CC * R.transpose(), rho, c, SS_pas_r, SS_act_r, SS_vol_r);

		CHECK_CLOSE(maxAbs(SS_pas_r - R * SS_pas * R.transpose()), 0.0, 1e-12);
		CHECK_CLOSE(maxAbs(SS_vol_r - R * SS_vol * R.transpose()), 0.0, 1e-12);
		CHECK_CLOSE(maxAbs(SS_act_r - R * SS_act * R.transpose()), 0.0, 1e-12);
	}
}

// 4. Uniaxial stretch along the fiber: the fiber-direction stress must pass
//    through zero at lamda = 1 and increase monotonically through it.
void checkUniaxialMonotonicity()
{
	std::cout << "  fiber stress is monotone in a uniaxial stretch\n";
	std::vector<double> gp = makeGlobalParameters();

	Vector3d a0(1., 0., 0.), s0(0., 1., 0.), n0(0., 0., 1.);
	const double kappa = 1. / 3., phif = 1.0;

	double previous = -std::numeric_limits<double>::infinity();
	for (double lamda : {0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30}) {
		Matrix3d CC = Matrix3d::Identity();
		CC(0, 0) = lamda * lamda;

		Matrix3d SS_pas, SS_act, SS_vol;
		evalSS(gp, phif, a0, s0, n0, kappa, 1., 1., 1., CC, /*rho*/ 0., /*c*/ 0.,
		       SS_pas, SS_act, SS_vol);

		double SS_aa = a0.dot((SS_pas + SS_vol) * a0);
		CHECK(SS_aa > previous);
		previous = SS_aa;

		if (std::fabs(lamda - 1.0) < 1e-14) {
			CHECK_CLOSE(SS_aa, 0.0, 1e-14);          // stress free at lamda = 1
		} else if (lamda > 1.0) {
			CHECK(SS_aa > 0.);                        // tension is positive
		} else {
			CHECK(SS_aa < 0.);                        // compression is negative
		}
	}
}

} // namespace

int main()
{
	std::cout << "=== kinematics and constitutive response ===\n";

	checkStressFreeReference();
	checkStressFreeGrownConfiguration();
	checkObjectivity();
	checkUniaxialMonotonicity();

	return testing::summary("kinematics");
}
