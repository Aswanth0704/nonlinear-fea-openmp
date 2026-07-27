/*
	Minimal header-only check harness so the test suite has no external dependency.
	Each test binary returns non-zero if any check failed, which is what CTest reads.
*/

#ifndef test_harness_h
#define test_harness_h

#include <cmath>
#include <iostream>
#include <string>

namespace testing {

inline int n_checks = 0;
inline int n_failures = 0;

inline void check(bool ok, const std::string &what, const char *file, int line)
{
	n_checks += 1;
	if (!ok) {
		n_failures += 1;
		std::cout << "  FAIL  " << what << "\n        at " << file << ":" << line << "\n";
	}
}

inline void check_close(double a, double b, double tol, const std::string &what, const char *file, int line)
{
	n_checks += 1;
	if (!(std::fabs(a - b) <= tol)) {
		n_failures += 1;
		std::cout << "  FAIL  " << what << "\n        " << a << " vs " << b
		          << "  (|diff| = " << std::fabs(a - b) << " > tol " << tol << ")"
		          << "\n        at " << file << ":" << line << "\n";
	}
}

inline int summary(const std::string &name)
{
	std::cout << name << ": " << (n_checks - n_failures) << "/" << n_checks << " checks passed\n";
	return n_failures == 0 ? 0 : 1;
}

} // namespace testing

#define CHECK(cond)             testing::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol)  testing::check_close((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)

#endif
