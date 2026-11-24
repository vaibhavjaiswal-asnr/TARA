
// Linux / ELF / GCC-or-Clang constructor interposer for NCrystal 4.4.6.
// This is a separate shared library; it does not change NCrystal or Liquids.
#include "NCrystal/interfaces/NCSABData.hh"
#include "NCrystal/internal/sabscatter/NCSABScatter.hh"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace NC = NCrystal;
namespace {
using Ctor = void (*)(NC::SABScatter*, NC::SABData&&, const NC::VectD&);
std::atomic<unsigned> next_id{0};

void write_number(std::ostream& out, double x)
{
  // Avoid silently producing invalid JSON if the model creates nonfinite values.
  if (!std::isfinite(x))
    throw std::runtime_error("non-finite number in SABData");
  out << x;
}

void write_binary_array(std::ostream& out, const NC::VectD& values)
{
  for (double value : values) {
    if (!std::isfinite(value))
      throw std::runtime_error("non-finite number in SABData");
  }
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(double)));
}

unsigned dump(const NC::SABData& d)
{
  const auto& a = d.alphaGrid();
  const auto& b = d.betaGrid();
  const auto& s = d.sab();
  if (a.empty() || b.empty() || s.size() != a.size() * b.size())
    throw std::runtime_error("inconsistent SABData dimensions");

  const char* dir = std::getenv("NCRYSTAL_SAB_DUMP_DIR");
  if (!dir || !*dir) throw std::runtime_error("missing capture directory");
  const unsigned id = next_id.fetch_add(1);
  static_assert(sizeof(double) == 8, "The SABData payload requires float64");
  std::ostringstream stem;
  stem << "sab_" << std::setfill('0') << std::setw(4) << id;
  const std::string binary_name = stem.str() + ".bin";
  const std::string binary_path = std::string(dir) + "/" + binary_name;
  std::ofstream binary(binary_path, std::ios::binary);
  if (!binary)
    throw std::runtime_error("could not open " + binary_path);
  // The reader expects alpha, beta, then contiguous beta rows of S.
  write_binary_array(binary, a);
  write_binary_array(binary, b);
  write_binary_array(binary, s);
  binary.close();
  if (!binary)
    throw std::runtime_error("failed writing " + binary_path);

  const std::string filename = std::string(dir) + "/" + stem.str() + ".json";
  std::ofstream out(filename);
  if (!out)
    throw std::runtime_error("could not open " + filename);
  out << std::setprecision(17);
  out << "{\n  \"temperature_K\": "; write_number(out, d.temperature().dbl());
  out << ",\n  \"bound_xs_barn\": "; write_number(out, d.boundXS().dbl());
  out << ",\n  \"element_mass_amu\": "; write_number(out, d.elementMassAMU().dbl());
  out << ",\n  \"suggested_emax_eV\": "; write_number(out, d.suggestedEmax());
  out << ",\n  \"binary_file\": \"" << binary_name << "\""
      << ",\n  \"n_alpha\": " << a.size()
      << ",\n  \"n_beta\": " << b.size()
      << ",\n  \"layout\": \"sab[beta_index][alpha_index]\"\n}\n";
  if (!out)
    throw std::runtime_error("failed writing " + filename);
  std::cerr << "[sabtap] exported " << filename
            << " (" << a.size() << " alpha x " << b.size() << " beta)\n";
  out.close();
  if (!out) throw std::runtime_error("failed closing capture");
  return id;
}

void forward(const char* sym, NC::SABScatter* self,
             NC::SABData&& sab, const NC::VectD& egrid)
{
  const char* dir = std::getenv("NCRYSTAL_SAB_DUMP_DIR");
  const bool capture = dir && *dir;
  unsigned id = 0;
  if (capture) id = dump(sab);

  dlerror();
  auto* addr = dlsym(RTLD_NEXT, sym);
  if (const char* error = dlerror()) {
    std::cerr << "[sabtap] cannot forward " << sym << ": " << error << '\n';
    std::abort();
  }
  reinterpret_cast<Ctor>(addr)(self, std::move(sab), egrid);
  if (capture) {
    // Use the constructed scatterer so this sidecar includes its native treatment.
    std::ifstream energies(std::string(dir) + "/energies.txt");
    if (!energies) throw std::runtime_error("missing energy grid");
    std::ostringstream name;
    name << dir << "/xs_" << std::setfill('0') << std::setw(4) << id << ".txt";
    std::ofstream xs(name.str());
    if (!xs) throw std::runtime_error("could not open " + name.str());
    xs << std::setprecision(17);
    NC::CachePtr cache;
    double energy;
    while (energies >> energy) {
      write_number(xs, self->crossSectionIsotropic(cache, NC::NeutronEnergy{energy}).dbl());
      xs << '\n';
    }
    xs.close();
    if (!xs) throw std::runtime_error("failed writing component cross sections");
  }

}
}

// Itanium C++ ABI symbols for NCrystal::SABScatter::SABScatter(
//     NCrystal::SABData&&, const std::vector<double>&).
// Both C1 (complete object) and C2 (base object) variants are covered.
// Verify against the exact installed libNCrystal using nm -D | c++filt.
extern "C" void sabtap_ctor_C1(NC::SABScatter*, NC::SABData&&, const NC::VectD&)
  __asm__("_ZN8NCrystal10SABScatterC1EONS_7SABDataERKSt6vectorIdSaIdEE");
extern "C" void sabtap_ctor_C2(NC::SABScatter*, NC::SABData&&, const NC::VectD&)
  __asm__("_ZN8NCrystal10SABScatterC2EONS_7SABDataERKSt6vectorIdSaIdEE");

extern "C" void sabtap_ctor_C1(NC::SABScatter* self, NC::SABData&& sab,
                               const NC::VectD& egrid)
{
  forward("_ZN8NCrystal10SABScatterC1EONS_7SABDataERKSt6vectorIdSaIdEE",
          self, std::move(sab), egrid);
}
extern "C" void sabtap_ctor_C2(NC::SABScatter* self, NC::SABData&& sab,
                               const NC::VectD& egrid)
{
  forward("_ZN8NCrystal10SABScatterC2EONS_7SABDataERKSt6vectorIdSaIdEE",
          self, std::move(sab), egrid);
}
