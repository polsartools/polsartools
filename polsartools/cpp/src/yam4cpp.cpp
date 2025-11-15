#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;
using cplx = std::complex<double>;

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>>
process_chunk_yam4cpp(std::vector<py::array_t<cplx>> chunks,
                      int window_size,
                      std::string model,
                      double SpanMin,
                      double SpanMax) {
    if (chunks.size() != 9)
        throw std::invalid_argument("Expected 9 complex chunks");

    auto shape = chunks[0].shape();
    int rows = shape[0];
    int cols = shape[1];

    py::array_t<double> M_odd({rows, cols});
    py::array_t<double> M_dbl({rows, cols});
    py::array_t<double> M_vol({rows, cols});
    py::array_t<double> M_hlx({rows, cols});

    auto M_odd_ptr = M_odd.mutable_unchecked<2>();
    auto M_dbl_ptr = M_dbl.mutable_unchecked<2>();
    auto M_vol_ptr = M_vol.mutable_unchecked<2>();
    auto M_hlx_ptr = M_hlx.mutable_unchecked<2>();

    std::vector<py::detail::unchecked_reference<cplx, 2>> chunk_refs;
    for (auto& chunk : chunks)
        chunk_refs.push_back(chunk.unchecked<2>());

    // Constants and small epsilon for floating-point comparison
    const double eps = 1e-6;
    
    for (int ii = 0; ii < rows; ++ii) {
        for (int jj = 0; jj < cols; ++jj) {
            std::vector<std::complex<double>> T3(9);  // Temporary array for storing 9 complex values
            for (int k = 0; k < 9; ++k)
                T3[k] = chunk_refs[k](ii, jj);  // Assign values from chunk_refs

            // Check if model is either "y4cr" or "y4cs" for unitary rotation
            if (model == "y4cr" || model == "y4cs") {
                // Calculate teta (angle) based on elements of T3
                double teta = 0.5 * std::atan2(2.0 * T3[5].real(), T3[4].real() - T3[8].real());

                // Apply unitary rotation on T3
                std::complex<double> T12 = T3[1], T13 = T3[2], T23 = T3[5];

                // Rotation on T3[1] and T3[2] (real and imaginary parts)
                T3[1] = T12.real() * std::cos(teta) + T13.real() * std::sin(teta)
                    + std::complex<double>(0, T12.imag() * std::cos(teta) + T13.imag() * std::sin(teta));
                T3[2] = -T12.real() * std::sin(teta) + T13.real() * std::cos(teta)
                    + std::complex<double>(0, -T12.imag() * std::sin(teta) + T13.imag() * std::cos(teta));

                // Update T3[4], T3[5], and T3[8] based on the rotation angle teta
                T3[4] = T3[4] * std::cos(teta) * std::cos(teta) + 2.0 * T23.real() * std::cos(teta) * std::sin(teta)
                    + T3[8] * std::sin(teta) * std::sin(teta);
                T3[5] = -T3[4] * std::cos(teta) * std::sin(teta) + T23.real() * (std::cos(teta) * std::cos(teta) - std::sin(teta) * std::sin(teta))
                    + T3[8] * std::cos(teta) * std::sin(teta) + std::complex<double>(0, T23.imag());
                T3[8] = T3[4] * std::sin(teta) * std::sin(teta) + T3[8] * std::cos(teta) * std::cos(teta)
                    - 2.0 * T23.real() * std::cos(teta) * std::sin(teta);
            }

            // Calculate total power (TP) and polarization components (Pc, Pv)
            double TP = T3[0].real() + T3[4].real() + T3[8].real();  // Total power is the sum of the real parts
            double Pc = 2.0 * std::abs(T3[5].imag());  // Polarization component

            double Pv = 0.0, Ps = 0.0, Pd = 0.0;  // Initialize the polarization components
            int HV_type = 1;  // Default to surface scattering model

            if (model == "y4cs") {
                // Calculate C1 as per the Python logic
                double C1 = T3[0].real() - T3[4].real() + (7.0 / 8.0) * T3[8].real() + (Pc / 16.0);
                
                // Determine HV_type based on C1
                HV_type = (C1 > 0.0) ? 1 : 2;  // Surface scattering (1) or Double bounce scattering (2)
            }

            // Surface scattering (HV_type == 1)
            if (HV_type == 1) {
                double ratio = 10.0 * std::log10((T3[0].real() + T3[4].real() - 2.0 * T3[1].real()) /
                                                (T3[0].real() + T3[4].real() + 2.0 * T3[1].real()));
                
                // Determine Pv based on the ratio
                if (-2.0 < ratio && ratio <= 2.0) {
                    Pv = 2.0 * (2.0 * T3[8].real() - Pc);
                } else {
                    Pv = (15.0 / 8.0) * (2.0 * T3[8].real() - Pc);
                }
            }

            // Double bounce scattering (HV_type == 2)
            if (HV_type == 2) {
                Pv = (15.0 / 16.0) * (2.0 * T3[8].real() - Pc);
            }


            // Freeman-Yamaguchi 3-components algorithm (Python translation)
                if (Pv < 0.0) {
                    double BETre = 0.0, BETim = 0.0, ALPre = 0.0, ALPim = 0.0, FS = 0.0, FD = 0.0, FV = 0.0;
                    double HHHH = (T3[0].real() + 2.0 * T3[1].real() + T3[4].real()) / 2.0;
                    double HHVVre = (T3[0].real() - T3[4].real()) / 2.0;
                    double HHVVim = -T3[1].imag();
                    double HVHV = T3[8].real() / 2.0;
                    double VVVV = (T3[0].real() - 2.0 * T3[1].real() + T3[4].real()) / 2.0;
                    
                    // Calculate the ratio for adjustments
                    double ratio = 10.0 * std::log10(VVVV / HHHH);

                    // Adjust FV, HHHH, VVVV, HHVVre based on the ratio
                    if (ratio <= -2.0) {
                        FV = 15.0 * (HVHV / 4.0);
                        HHHH -= 8.0 * (FV / 15.0);
                        VVVV -= 3.0 * (FV / 15.0);
                        HHVVre -= 2.0 * (FV / 15.0);
                    }
                    if (ratio > 2.0) {
                        FV = 15.0 * (HVHV / 4.0);
                        HHHH -= 3.0 * (FV / 15.0);
                        VVVV -= 8.0 * (FV / 15.0);
                        HHVVre -= 2.0 * (FV / 15.0);
                    }
                    if (-2.0 < ratio && ratio <= 2.0) {
                        FV = 8.0 * (HVHV / 2.0);
                        HHHH -= 3.0 * (FV / 8.0);
                        VVVV -= 3.0 * (FV / 8.0);
                        HHVVre -= 1.0 * (FV / 8.0);
                    }

                    // Case 1: Volume Scatter > Total
                    if (HHHH <= eps || VVVV <= eps) {
                        FD = 0.0;
                        FS = 0.0;
                        if (-2.0 < ratio && ratio <= 2.0) {
                            FV = (HHHH + 3.0 * (FV / 8.0)) + HVHV + (VVVV + 3.0 * (FV / 8.0));
                        }
                        if (ratio <= -2.0) {
                            FV = (HHHH + 8.0 * (FV / 15.0)) + HVHV + (VVVV + 3.0 * (FV / 15.0));
                        }
                        if (ratio > 2.0) {
                            FV = (HHHH + 3.0 * (FV / 15.0)) + HVHV + (VVVV + 8.0 * (FV / 15.0));
                        }
                    } else {
                        // Data conditioning for non-realizable ShhSvv* term
                        double rtemp = HHVVre * HHVVre + HHVVim * HHVVim;
                        if (rtemp > HHHH * VVVV) {
                            HHVVre = HHVVre * std::sqrt((HHHH * VVVV) / rtemp);
                            HHVVim = HHVVim * std::sqrt((HHHH * VVVV) / rtemp);
                        }

                        // Odd Bounce
                        if (HHVVre >= 0.0) {
                            ALPre = -1.0;
                            ALPim = 0.0;
                            FD = (HHHH * VVVV - HHVVre * HHVVre - HHVVim * HHVVim) / (HHHH + VVVV + 2.0 * HHVVre);
                            FS = VVVV - FD;
                            BETre = (FD + HHVVre) / FS;
                            BETim = HHVVim / FS;
                        }

                        // Even Bounce
                        if (HHVVre < 0.0) {
                            BETre = 1.0;
                            BETim = 0.0;
                            FS = (HHHH * VVVV - HHVVre * HHVVre - HHVVim * HHVVim) / (HHHH + VVVV - 2.0 * HHVVre);
                            FD = VVVV - FS;
                            ALPre = (HHVVre - FS) / FD;
                            ALPim = HHVVim / FD;
                        }
                    }

                    // Store results in the matrices
                    M_odd_ptr(ii, jj) = FS * (1 + BETre * BETre + BETim * BETim);
                    M_dbl_ptr(ii, jj) = FD * (1 + ALPre * ALPre + ALPim * ALPim);
                    M_vol_ptr(ii, jj) = FV;
                    M_hlx_ptr(ii, jj) = 0.0;

                    // Apply Span limits
                    M_odd_ptr(ii, jj) = std::clamp(M_odd_ptr(ii, jj), SpanMin, SpanMax);
                    M_dbl_ptr(ii, jj) = std::clamp(M_dbl_ptr(ii, jj), SpanMin, SpanMax);
                    M_vol_ptr(ii, jj) = std::clamp(M_vol_ptr(ii, jj), SpanMin, SpanMax);
                }
            else {
                // Surface scattering (HV_type == 1)
                if (HV_type == 1) {
                    double S = T3[0].real() - (Pv / 2.0);
                    double D = TP - Pv - Pc - S;
                    double Cre = T3[1].real() + T3[2].real();
                    double Cim = T3[1].imag() + T3[2].imag();
                    double ratio = 10.0 * std::log10((T3[0].real() + T3[4].real() - 2.0 * T3[1].real()) /
                                                                    (T3[0].real() + T3[4].real() + 2.0 * T3[1].real()));
                                    
                    // Adjust Cre based on ratio (this logic matches the Python code)
                    if (ratio <= -2.0) {
                        Cre -= (Pv / 6.0);
                    }
                    if (ratio > 2.0) {
                        Cre += (Pv / 6.0);
                    }

                    // Check and adjust Pv, Pc, Ps, and Pd values as per the conditions
                    if ((Pv + Pc) > TP) {
                        Ps = 0.0;
                        Pd = 0.0;
                        Pv = TP - Pc;
                    } else {
                        double CO = 2.0 * T3[0].real() + Pc - TP;
                        if (CO > 0.0) {
                            Ps = S + ((Cre * Cre + Cim * Cim) / S);
                            Pd = D - ((Cre * Cre + Cim * Cim) / S);
                        } else {
                            Pd = D + ((Cre * Cre + Cim * Cim) / D);
                            Ps = S - ((Cre * Cre + Cim * Cim) / D);
                        }
                    }

                    // Ensure Ps and Pd are non-negative and adjust as needed
                    if (Ps < 0.0) {
                        if (Pd < 0.0) {
                            Ps = 0.0;
                            Pd = 0.0;
                            Pv = TP - Pc;
                        } else {
                            Ps = 0.0;
                            Pd = TP - Pv - Pc;
                        }
                    } else {
                        if (Pd < 0.0) {
                            Pd = 0.0;
                            Ps = TP - Pv - Pc;
                        }
                    }
                }
                // Double bounce scattering (HV_type == 2)
                else if (HV_type == 2) {
                    double S = T3[0].real();
                    double D = TP - Pv - Pc - S;

                    double Cre = T3[1].real() + T3[2].real();
                    double Cim = T3[1].imag() + T3[2].imag();

                    // For double bounce, calculate Ps and Pd
                    Pd = D + ((Cre * Cre + Cim * Cim) / D);
                    Ps = S - ((Cre * Cre + Cim * Cim) / D);

                    // Ensure Ps and Pd are non-negative and adjust as needed
                    if (Ps < 0.0) {
                        if (Pd < 0.0) {
                            Ps = 0.0;
                            Pd = 0.0;
                            Pv = TP - Pc;
                        } else {
                            Ps = 0.0;
                            Pd = TP - Pv - Pc;
                        }
                    } else {
                        if (Pd < 0.0) {
                            Pd = 0.0;
                            Ps = TP - Pv - Pc;
                        }
                    }
                }

                // Ensure all components are non-negative and within bounds (matching Python logic)
                if (Ps < 0.0) {
                    Ps = 0.0;
                }
                if (Pd < 0.0) {
                    Pd = 0.0;
                }
                if (Pv < 0.0) {
                    Pv = 0.0;
                }
                if (Pc < 0.0) {
                    Pc = 0.0;
                }

                // Ensure all components are within the specified range [SpanMin, SpanMax]
                M_odd_ptr(ii, jj) = std::clamp(Ps, 0.0, SpanMax);
                M_dbl_ptr(ii, jj) = std::clamp(Pd, 0.0, SpanMax);
                M_vol_ptr(ii, jj) = std::clamp(Pv, 0.0, SpanMax);
                M_hlx_ptr(ii, jj) = std::clamp(Pc, 0.0, SpanMax);
                // M_hlx_ptr(ii, jj) = Pc;
            }
        }
    }

    return std::make_tuple(M_odd, M_dbl, M_vol, M_hlx);
}
PYBIND11_MODULE(yam4cpp, m) {
    m.def("process_chunk_yam4cpp", &process_chunk_yam4cpp,
          "Process SAR matrix chunks and return scattering powers");
}
