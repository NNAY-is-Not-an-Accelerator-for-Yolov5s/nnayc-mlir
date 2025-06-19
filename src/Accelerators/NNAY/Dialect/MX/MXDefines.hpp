#ifndef MX_DEFINES_HPP
#define MX_DEFINES_HPP

#include <cassert> // For assert
#include <cmath>   // For std::pow for verification prints
#include <iomanip> // For std::hex, std::dec
#include <iostream>
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For std::runtime_error
#include <vector>

// Forward declarations
template <int MantissaStoredBits, int ExponentStoredBits>
class FP;
template <int S_Width, int M_Width>
class MX;
template <int S_Ext_Width, int M_Ext_Width>
class MX_ext;

// Helper to get number of bits for unsigned long long
constexpr int ULL_BITS = sizeof(unsigned long long) * 8;

/**
 * Class FP: Simulates a floating-point number.
 * Value = (-1)^sign * mantissa * 2^exponent
 */
template <int MantissaStoredBits = ULL_BITS, int ExponentStoredBits = 64>
class FP {
public:
  bool sign;
  long long exponent;
  unsigned long long mantissa;

  static constexpr int MANTISSA_CAPACITY_BITS = MantissaStoredBits;
  static constexpr int EXPONENT_CAPACITY_BITS = ExponentStoredBits;

  FP(bool s = false, long long exp = 0, unsigned long long mant = 0ULL)
      : sign(s), exponent(exp), mantissa(mant) {
    normalize();
  }

  void normalize() {
    if (mantissa == 0) {
      exponent = 0;
      return;
    }

    unsigned int shifts_needed = 0;
    if (mantissa > 0) {
      unsigned long long temp_m = mantissa;
      unsigned long long msb_ull = 1ULL << (ULL_BITS - 1);
      while (!(temp_m & msb_ull) && shifts_needed < ULL_BITS) {
        temp_m <<= 1;
        shifts_needed++;
        if (temp_m == 0 && mantissa != 0) {
          shifts_needed = ULL_BITS;
          break;
        }
      }
    } else {
      exponent = 0;
      return;
    }

    if (shifts_needed > 0 && shifts_needed < ULL_BITS) {
      mantissa <<= shifts_needed;
      exponent -= shifts_needed;
    }
  }

  FP operator+(const FP &other) const {
    if (this->mantissa == 0)
      return other;
    if (other.mantissa == 0)
      return *this;

    FP result;

    unsigned long long m1_aligned = this->mantissa;
    long long e1 = this->exponent;
    bool s1 = this->sign;

    unsigned long long m2_aligned = other.mantissa;
    long long e2 = other.exponent;
    bool s2 = other.sign;

    long long common_exponent;

    if (e1 > e2) {
      long long shift = e1 - e2;
      common_exponent = e1;
      if (shift < ULL_BITS)
        m2_aligned >>= shift;
      else
        m2_aligned = 0;
    } else if (e2 > e1) {
      long long shift = e2 - e1;
      common_exponent = e2;
      if (shift < ULL_BITS)
        m1_aligned >>= shift;
      else
        m1_aligned = 0;
    } else {
      common_exponent = e1;
    }

    result.exponent = common_exponent;

    if (s1 == s2) {
      result.sign = s1;
      unsigned long long sum = m1_aligned + m2_aligned;
      if (sum < m1_aligned && m1_aligned > 0) {
        result.mantissa = (m1_aligned >> 1) + (m2_aligned >> 1) +
                          ((m1_aligned | m2_aligned) & 1);
        result.exponent++;
      } else {
        result.mantissa = sum;
      }
    } else {
      if (m1_aligned >= m2_aligned) {
        result.mantissa = m1_aligned - m2_aligned;
        result.sign = s1;
      } else {
        result.mantissa = m2_aligned - m1_aligned;
        result.sign = s2;
      }
    }

    result.normalize();
    return result;
  }

  double get_value() const {
    if (mantissa == 0)
      return 0.0;
    // The stored normalized mantissa and exponent directly give the value:
    // mantissa_normalized * 2^exponent_normalized
    // where mantissa_normalized is an integer and exponent_normalized is its
    // corresponding power of 2. E.g. 5 (0b101) * 2^0. If ULL_BITS=64, norm_mant
    // = 0b101 << 61, norm_exp = 0 - 61. val = (0b101 << 61) * पॉव(2, -61) =
    // 0b101 * 2^61 * 2^-61 = 5.
    return (sign ? -1.0 : 1.0) * static_cast<double>(mantissa) *
           std::pow(2.0, exponent);
  }

  void print() const {
    std::cout << "FP(sign=" << sign << ", exp=" << exponent << ", mant=0x"
              << std::hex << std::setw(16) << std::setfill('0') << mantissa
              << std::dec << ") /* approx val: " << get_value() << " */"
              << std::endl;
  }
};

/**
 * Class MX: Number representation.
 * s field is biased by 127.
 * Value = (-1)^signbit * m_unsigned_value * 2^((s_field - bias) - ss + m_width
 * - 1)
 */
template <int S_Width, int M_Width>
class MX {
public:
  long long s_val; // Biased exponent value
  bool ss_val;
  bool signbit_val;
  unsigned long long m_val;

  static constexpr int s_width = S_Width; // Conceptual bit width
  static constexpr int m_width = M_Width; // Actual bit width for m_val masking
  static constexpr int exponent_bias = 127;

  MX(long long s_field = 0, bool ss = false, bool signbit = false,
      unsigned long long m = 0ULL)
      : s_val(s_field), ss_val(ss), signbit_val(signbit) {
    if (M_Width > 0 && M_Width < ULL_BITS) {
      unsigned long long mask = (1ULL << M_Width) - 1;
      this->m_val = m & mask;
    } else if (M_Width == 0) {
      this->m_val = 0;
    } else { // M_Width >= ULL_BITS or unspecified negative (treat as ULL_BITS)
      this->m_val = m;
    }
    // s_val is not currently clamped by S_Width in this implementation.
  }

  bool get_signbit() const { return signbit_val; }
  unsigned long long get_m_unsigned_value() const { return m_val; }
  bool get_ss() const { return ss_val; }
  long long get_s_field() const { return s_val; }

  double calculate_value() const {
    double m_component = static_cast<double>(m_val);
    double effective_s = static_cast<double>(s_val) - exponent_bias;
    double exp_val = effective_s + (ss_val ? -1.0 : 0.0) -
                     static_cast<double>(m_width) + 1.0;
    return (signbit_val ? -1.0 : 1.0) * m_component * std::pow(2.0, exp_val);
  }

  void print() const {
    std::cout << "MX(s=" << s_val << "(biased_s=" << s_val - exponent_bias
              << ")"
              << ", ss=" << ss_val << ", sign=" << signbit_val
              << ", m=" << m_val << " [m_w=" << m_width << ", s_w=" << s_width
              << "])"
              << " /* val: " << calculate_value() << " */" << std::endl;
  }
};

/**
 * Class MX_ext: Number representation.
 * s field is biased by 127.
 * Value = (-1)^signbit * m_unsigned_value * 2^((s_field - bias) + m_ext_width -
 * 1)
 */
template <int S_Ext_Width, int M_Ext_Width>
class MX_ext {
public:
  long long s_val; // Biased exponent value
  bool signbit_val;
  unsigned long long m_val;
  long fixed_point_val; // Fixed point value for m_val

  static constexpr int s_ext_width = S_Ext_Width; // Conceptual bit width
  static constexpr int m_ext_width =
      M_Ext_Width; // Actual bit width for m_val masking
  static constexpr int exponent_bias = 127;

  MX_ext(long long s_field = 0, bool signbit = false,
      unsigned long long m = 0ULL, long fixed_point_val = 1)
      : s_val(s_field), signbit_val(signbit), fixed_point_val(fixed_point_val) {
    if (M_Ext_Width > 0 && M_Ext_Width < ULL_BITS) {
      unsigned long long mask = (1ULL << M_Ext_Width) - 1;
      this->m_val = m & mask;
    } else if (M_Ext_Width == 0) {
      this->m_val = 0;
    } else { // M_Ext_Width >= ULL_BITS or unspecified negative (treat as
             // ULL_BITS)
      this->m_val = m;
    }
    // s_val is not currently clamped by S_Ext_Width in this implementation.
  }

  bool get_signbit() const { return signbit_val; }
  unsigned long long get_m_unsigned_value() const { return m_val; }
  long long get_s_field() const { return s_val; }
  long get_fixed_point_val() const { return fixed_point_val; }

  double calculate_value() const {
    double m_component = static_cast<double>(m_val);
    double effective_s = static_cast<double>(s_val) - exponent_bias;
    double exp_val =
        effective_s - (static_cast<double>(m_ext_width) - fixed_point_val);
    return (signbit_val ? -1.0 : 1.0) * m_component * std::pow(2.0, exp_val);
  }

  void print() const {
    std::cout << "MX_ext(s_fld=" << s_val
              << "(u_s_contrib=" << s_val - exponent_bias << ")"
              << ", sign=" << signbit_val << ", m=" << m_val
              << " [m_ext_w=" << m_ext_width << ", s_ext_w=" << s_ext_width
              << "])"
              << " /* val: " << calculate_value() << " */" << std::endl;
  }
};

// Conversion Functions

template <int S_Width_Orig, int M_Width_Orig>
MX_ext<S_Width_Orig, M_Width_Orig + 1> convert_mx_to_mx_ext(
    const MX<S_Width_Orig, M_Width_Orig> &mx) {
  unsigned long long mx_m_val = mx.get_m_unsigned_value();
  unsigned long long new_mx_m_val;
  bool mx_ss = mx.get_ss();
  long long s_field = mx.get_s_field();

  if (mx_ss) {
    // Nothing needs to be done
    new_mx_m_val = mx_m_val;
  } else {
    new_mx_m_val = mx_m_val << 1;
  }

  return MX_ext<S_Width_Orig, M_Width_Orig + 1>(
      s_field, mx.get_signbit(), new_mx_m_val);
}

template <int S_Ext_Width_Orig, int M_Ext_Width_Orig, int S_Width_New>
MX<S_Width_New, M_Ext_Width_Orig> convert_mx_ext_to_mx(
    const MX_ext<S_Ext_Width_Orig, M_Ext_Width_Orig> &ext) {
  unsigned long long ext_m_val = ext.get_m_unsigned_value();
  long long ext_s_field_val = ext.get_s_field();
  bool recovered_ss = false;

  if (ext_m_val == 0) {
    recovered_ss = false;
  } else if (ext_s_field_val == 0) {
    if (ext_m_val == 1) {
      recovered_ss = true;
    } else {
      recovered_ss = false;
    }
  } else {
    if (ext_m_val % ext_s_field_val == 0) {
      unsigned long long ratio = ext_m_val / ext_s_field_val;
      if (ratio == 1) {
        recovered_ss = false;
      } else if (ratio == 2 && (ext_m_val % 2 == 0) &&
                 (ext_m_val / 2 == ext_s_field_val)) {
        recovered_ss = true;
      } else {
        recovered_ss = false;
      }
    } else {
      recovered_ss = false;
    }
  }

  long long new_mx_s_field = ext_s_field_val + (recovered_ss ? 1 : 0);
  return MX<S_Width_New, M_Ext_Width_Orig>(
      new_mx_s_field, recovered_ss, ext.get_signbit(), ext_m_val);
}

template <int S_Ext_Width_Orig, int M_Ext_Width_Orig,
    int FP_MantissaBits = ULL_BITS, int FP_ExponentBits = 64>
FP<FP_MantissaBits, FP_ExponentBits> convert_mx_ext_to_fp(
    const MX_ext<S_Ext_Width_Orig, M_Ext_Width_Orig> &ext) {
  bool fp_sign = ext.get_signbit();
  unsigned long long fp_mant_val = ext.get_m_unsigned_value();
  long long ext_s_field = ext.get_s_field();

  long long unbiased_s =
      ext_s_field - MX_ext<S_Ext_Width_Orig, M_Ext_Width_Orig>::exponent_bias;
  long long fp_exp_val =
      unbiased_s +
      (static_cast<long long>(
           MX_ext<S_Ext_Width_Orig, M_Ext_Width_Orig>::m_ext_width) -
          1);

  return FP<FP_MantissaBits, FP_ExponentBits>(fp_sign, fp_exp_val, fp_mant_val);
}

// New MX_ext operations

/**
 * Multiplication of two MX_ext numbers.
 * Output s_field = s1_field + s2_field
 * Output m_val = m1_val * m2_val
 * Output signbit = sign1 ^ sign2
 */
template <int S_Ext_W, int M_W>
MX_ext<S_Ext_W + 1, M_W * 2> multiply_mx_ext(
    const MX_ext<S_Ext_W, M_W> &op1, const MX_ext<S_Ext_W, M_W> &op2) {
  long long res_s_field = op1.get_s_field() + op2.get_s_field() -
                          MX_ext<S_Ext_W, M_W>::exponent_bias;
  unsigned long long res_m_val =
      op1.get_m_unsigned_value() * op2.get_m_unsigned_value();
  bool res_signbit = op1.get_signbit() ^ op2.get_signbit();
  long res_fixed_point_val =
      op1.get_fixed_point_val() + op2.get_fixed_point_val();

  return MX_ext<S_Ext_W + 1, M_W * 2>(
      res_s_field, res_signbit, res_m_val, res_fixed_point_val);
}

/**
 * Special addition of two MX_ext numbers.
 * Requires op1.s_field == op2.s_field.
 * Handles different sign bits by performing subtraction of m_vals.
 * Output s_field = op1.s_field (common s_field).
 * Output m_val depends on sign comparison.
 * Output signbit depends on sign comparison and magnitude of m_vals.
 */

template <int S_Ext_W,
    int M_W> // M_W is M_Ext_Width from the input MX_ext types
MX_ext<S_Ext_W, M_W + 1> special_add_mx_ext(
    const MX_ext<S_Ext_W, M_W> &op1, const MX_ext<S_Ext_W, M_W> &op2) {
  if (op1.get_s_field() != op2.get_s_field()) {
    throw std::runtime_error(
        "Special MX_ext addition requires s_fields to be equal.");
  }

  long long res_s_field = op1.get_s_field();
  unsigned long long m1_val = op1.get_m_unsigned_value();
  bool sign1 = op1.get_signbit();
  unsigned long long m2_val = op2.get_m_unsigned_value();
  bool sign2 = op2.get_signbit();

  assert(op1.fixed_point_val ==
         op2.fixed_point_val); // Ensure fixed point values are the same
  long res_fixed_point_val = op1.get_fixed_point_val() + 1;

  unsigned long long res_m_val;
  bool res_signbit;

  if (sign1 == sign2) {
    // Signs are the same, perform addition
    res_m_val = m1_val + m2_val;
    res_signbit = sign1; // or sign2, they are the same
  } else {
    // Signs are different, perform subtraction
    if (m1_val >= m2_val) {
      res_m_val = m1_val - m2_val;
      res_signbit = sign1; // Sign of the larger absolute value
    } else {
      res_m_val = m2_val - m1_val;
      res_signbit = sign2; // Sign of the larger absolute value
    }
  }

  // The MX_ext constructor will handle masking res_m_val to M_W + 1 bits.
  return MX_ext<S_Ext_W, M_W + 1>(
      res_s_field, res_signbit, res_m_val, res_fixed_point_val);
}

// Helper to pre-process inputs: replace NaN/Inf with 0.0
// This mimics the nan_to_num_(0.0, 0.0, 0.0) behavior.
std::vector<float> inline preprocess_fp32_inputs_for_quantization(
    const std::vector<float> &inputs) {
  std::vector<float> processed_inputs = inputs;
  for (float &val : processed_inputs) {
    if (std::isnan(val) || std::isinf(val)) {
      val = 0.0f;
    }
  }
  return processed_inputs;
}

/**
 * @brief Quantizes a vector of 16 FP32 numbers to 16 MX<S_Width, M_Width>
 * numbers.
 *
 * @tparam S_Width The conceptual bit width of the 's' field in the output MX
 * type.
 * @tparam M_Width The bit width of the 'm' field in the output MX type.
 * @param fp32_inputs_original A vector of 16 float values.
 * @return A vector of 16 MX<S_Width, M_Width> quantized values.
 * @throws std::runtime_error if input vector size is not 16.
 */
template <int S_Width, int M_Width>
std::vector<MX<S_Width, M_Width>> quantize_fp32_to_mx_vector(
    const std::vector<float> &fp32_inputs_original) {
  if (fp32_inputs_original.size() != 16) {
    throw std::runtime_error(
        "Input vector size must be 16 for quantize_fp32_to_mx_vector.");
  }

  std::vector<float> fp32_inputs =
      preprocess_fp32_inputs_for_quantization(fp32_inputs_original);

  std::vector<int> actual_exponents(16);
  std::vector<bool> signs(16);
  int max_actual_exp = std::numeric_limits<int>::min();
  bool any_nonzero_input = false;

  for (int i = 0; i < 16; ++i) {
    float current_fp = fp32_inputs[i];
    signs[i] = std::signbit(current_fp);
    float abs_fp = std::abs(current_fp);

    if (abs_fp > 0.0f) {
      any_nonzero_input = true;
      actual_exponents[i] = static_cast<int>(std::floor(std::log2(abs_fp)));
      if (actual_exponents[i] > max_actual_exp) {
        max_actual_exp = actual_exponents[i];
      }
    } else {
      actual_exponents[i] = std::numeric_limits<int>::min() + 1;
    }
  }

  if (!any_nonzero_input) {
    // Default exponent if all inputs are zero (e.g., FP32 min normal exponent
    // bias adjusted)
    max_actual_exp =
        -MX<S_Width, M_Width>::exponent_bias; // Results in s_field of 0
  }

  // Clamp max_actual_exp to a reasonable range if necessary, e.g. FP32 limits.
  // Max possible exponent in standard FP32 is 127.
  if (max_actual_exp > 127)
    max_actual_exp = 127;
  // Min normal exponent for FP32 is -126. Denormals can be lower.
  // If all inputs were denormal, max_actual_exp could be < -126.

  long long tentative_s_field = static_cast<long long>(max_actual_exp) +
                                MX<S_Width, M_Width>::exponent_bias;
  long long common_s_field;

  // Clamping for s_field based on S_Width (assuming s_field is conceptually
  // unsigned-like for its range) The MX class stores s_val as long long, but
  // this quantization limits its value.
  unsigned long long s_field_unsigned_max = 0;
  if (S_Width > 0 && S_Width < (sizeof(unsigned long long) *
                                   8)) { // Avoid overflow for 1ULL << S_Width
    s_field_unsigned_max = (1ULL << S_Width) - 1;
  } else if (S_Width >= (sizeof(unsigned long long) *
                            8)) { // Max S_Width is effectively ULLONG_MAX
    s_field_unsigned_max = std::numeric_limits<unsigned long long>::max();
  }
  // S_Width == 0 is not typical for an exponent field, but if so, max would be
  // 0.

  if (tentative_s_field < 0) {
    common_s_field = 0;
    // It's unusual for a biased exponent field to be negative if bias is ~127
    // and S_Width is e.g. 8 This implies max_actual_exp was very small (e.g.
    // <-127)
    // std::cerr << "Warning: Quantization determined max_actual_exp ("
    //           << max_actual_exp << ") leading to a biased s_field ("
    //           << tentative_s_field
    //           << ") that is negative. Clamping common_s_field to 0."
    //           << std::endl;
  } else if (static_cast<unsigned long long>(tentative_s_field) >
                 s_field_unsigned_max &&
             S_Width > 0) {
    common_s_field = static_cast<long long>(s_field_unsigned_max);
    // std::cerr << "Warning: Quantization determined max_actual_exp ("
    //           << max_actual_exp << ") leading to a biased s_field ("
    //           << tentative_s_field << ") exceeding S_Width (" << S_Width
    //           << " bits) capacity (" << s_field_unsigned_max
    //           << "). Clamping common_s_field to max value." << std::endl;
  } else {
    common_s_field = tentative_s_field;
  }
  if (S_Width == 0) { // If S_Width is 0, s_field must be 0
    common_s_field = 0;
  }

  std::vector<MX<S_Width, M_Width>> mx_outputs(16);
  unsigned long long m_quant_max = 0;
  if (M_Width > 0 && M_Width < (sizeof(unsigned long long) * 8)) {
    m_quant_max = (1ULL << M_Width) - 1;
  } else if (M_Width >= (sizeof(unsigned long long) * 8)) {
    m_quant_max = std::numeric_limits<unsigned long long>::max();
  }
  // If M_Width is 0, m_quant_max remains 0.

  for (int group_idx = 0; group_idx < 8; ++group_idx) {
    int i1 = group_idx * 2;
    int i2 = group_idx * 2 + 1;

    float abs_val1 = std::abs(fp32_inputs[i1]);
    int exp1 = (abs_val1 > 0.0f) ? actual_exponents[i1]
                                 : (std::numeric_limits<int>::min() + 1);

    float abs_val2 = std::abs(fp32_inputs[i2]);
    int exp2 = (abs_val2 > 0.0f) ? actual_exponents[i2]
                                 : (std::numeric_limits<int>::min() + 1);

    bool exp_cond1 = (abs_val1 > 0.0f) ? (max_actual_exp - exp1 >= 1) : true;
    bool exp_cond2 = (abs_val2 > 0.0f) ? (max_actual_exp - exp2 >= 1) : true;
    bool common_ss_val = (exp_cond1 && exp_cond2);

    // Corrected m_scale_factor expression
    double m_scale_factor_val = static_cast<double>(max_actual_exp) -
                                (common_ss_val ? 1.0 : 0.0) -
                                (static_cast<double>(M_Width) - 1.0);
    double m_scale_factor = std::pow(2.0, m_scale_factor_val);

    unsigned long long m_val1 = 0;
    if (abs_val1 > 0.0f && M_Width > 0) {
      if (m_scale_factor >
          std::numeric_limits<double>::epsilon()) { // Avoid division by
                                                    // zero/small
        double m_double = abs_val1 / m_scale_factor;
        unsigned long long m_temp_val =
            static_cast<unsigned long long>(std::round(m_double));
        if (m_temp_val > m_quant_max) {
          m_val1 = m_quant_max;
          // std::cerr << "Warning: Input val " << fp32_inputs_original[i1]
          //           << " (abs=" << abs_val1 << ")"
          //           << " results in m_val (" << m_temp_val
          //           << ") exceeding M_Width (" << M_Width << " bits) capacity ("
          //           << m_quant_max
          //           << "). Clamping to max value. Scale factor exp: "
          //           << m_scale_factor_val << std::endl;
        } else {
          m_val1 = m_temp_val;
        }
      } else { // Scale factor is zero or too small
               // If abs_val1 is non-zero, it means m_val should be max.
               // If abs_val1 is also zero, m_val remains 0.
        m_val1 = m_quant_max;
        // std::cerr << "Warning: Input val " << fp32_inputs_original[i1]
        //           << " (abs=" << abs_val1 << ")"
        //           << " with very small/zero m_scale_factor (exp: "
        //           << m_scale_factor_val << "). Clamping m_val to "
        //           << m_quant_max << "." << std::endl;
      }
    }
    mx_outputs[i1] =
        MX<S_Width, M_Width>(common_s_field, common_ss_val, signs[i1], m_val1);

    unsigned long long m_val2 = 0;
    if (abs_val2 > 0.0f && M_Width > 0) {
      // m_scale_factor is the same for the pair
      if (m_scale_factor > std::numeric_limits<double>::epsilon()) {
        double m_double = abs_val2 / m_scale_factor;
        unsigned long long m_temp_val =
            static_cast<unsigned long long>(std::round(m_double));
        if (m_temp_val > m_quant_max) {
          m_val2 = m_quant_max;
          // std::cerr << "Warning: Input val " << fp32_inputs_original[i2]
          //           << " (abs=" << abs_val2 << ")"
          //           << " results in m_val (" << m_temp_val
          //           << ") exceeding M_Width (" << M_Width << " bits) capacity ("
          //           << m_quant_max
          //           << "). Clamping to max value. Scale factor exp: "
          //           << m_scale_factor_val << std::endl;
        } else {
          m_val2 = m_temp_val;
        }
      } else {
        m_val2 = m_quant_max;
        // std::cerr << "Warning: Input val " << fp32_inputs_original[i2]
        //           << " (abs=" << abs_val2 << ")"
        //           << " with very small/zero m_scale_factor (exp: "
        //           << m_scale_factor_val << "). Clamping m_val to "
        //           << m_quant_max << "." << std::endl;
      }
    }
    mx_outputs[i2] =
        MX<S_Width, M_Width>(common_s_field, common_ss_val, signs[i2], m_val2);
  }
  return mx_outputs;
}

/**
 * Processes two groups of 16 MX numbers.
 * 1. Converts MX to MX_ext.
 * 2. Performs element-wise multiplication of the two groups of MX_ext numbers.
 * 3. Sums the 16 product terms using an addition tree with specified
 * intermediate MX_ext types.
 *
 * @tparam S_W The S_Width of the input MX numbers.
 * @tparam M_W The M_Width of the input MX numbers.
 * @param group1_mx First group of 16 MX numbers.
 * @param group2_mx Second group of 16 MX numbers.
 * @return The final sum as an MX_ext<2*S_W, 2*M_W+4> number.
 * @throws std::runtime_error if input groups are not size 16, or if s_fields
 * mismatch during addition.
 */
template <int S_W, int M_W>
MX_ext<S_W + 1, 2 * M_W + 6> process_mx_groups_to_final_sum(
    const std::vector<MX<S_W, M_W>> &group1_mx,
    const std::vector<MX<S_W, M_W>> &group2_mx) {

  if (group1_mx.size() != 16 || group2_mx.size() != 16) {
    throw std::runtime_error("Input groups must each contain 16 MX numbers.");
  }

  // Step 2: Convert MX to MX_ext
  // MX<S_W, M_W> -> MX_ext<S_W, M_W+1>
  std::vector<MX_ext<S_W, M_W + 1>> group1_ext(16);
  std::vector<MX_ext<S_W, M_W + 1>> group2_ext(16);

  for (size_t i = 0; i < 16; ++i) {
    group1_ext[i] = convert_mx_to_mx_ext(group1_mx[i]);
    group2_ext[i] = convert_mx_to_mx_ext(group2_mx[i]);
    // std::cout << "Group 1: " << std::endl; // Debug print
    // group1_ext[i].print(); // Debug print
    // std::cout << "Group 2: " << std::endl; // Debug print
    // group2_ext[i].print(); // Debug print
    // std::cout << std::endl; // Debug print
  }

  // Step 3: Element-wise multiplication
  // Input to multiply_mx_ext is MX_ext<S_W, M_W+1>.
  // Output is MX_ext<S_W+1, (M_W+1)*2>.
  // Let S_Prod = S_W+1 and M_Prod = (M_W+1)*2 = 2*M_W+2.
  constexpr int S_Prod = S_W + 1;
  constexpr int M_Prod = (M_W + 1) * 2;
  std::vector<MX_ext<S_Prod, M_Prod>> products(16);

  for (size_t i = 0; i < 16; ++i) {
    products[i] = multiply_mx_ext(group1_ext[i], group2_ext[i]);
    // std::cout << "OP 1: " << std::endl; // Debug print
    // group1_ext[i].print();
    // std::cout << "OP 2: " << std::endl; // Debug print
    // group2_ext[i].print();
    // std::cout << "Product: " << std::endl; // Debug print
    // products[i].print(); // Debug print
    // std::cout << std::endl;
  }

  // Step 4: Addition Tree
  // The S-width for all addition tree stages' outputs is specified as 2*S_W
  // (which is S_Prod).

  // Stage 1 (16 -> 8): Sum pairs of products.
  // Input type: MX_ext<S_Prod, M_Prod>
  // Output type: MX_ext<S_Prod, 2*M_W+1>
  constexpr int M_Stage1 = M_Prod;
  std::vector<MX_ext<S_Prod, M_Stage1 + 1>> stage1_sums(8);
  for (size_t i = 0; i < 8; ++i) {
    stage1_sums[i] = special_add_mx_ext<S_Prod, M_Stage1>(
        products[2 * i], products[2 * i + 1]);
    // std::cout << "OP 1 : " << std::endl; // Debug print
    // products[2 * i].print();
    // std::cout << "OP 2 : " << std::endl; // Debug print
    // products[2 * i + 1].print();
    // std::cout << "Sum : " << std::endl; // Debug print
    // stage1_sums[i].print(); // Debug print
  }

  // Stage 2 (8 -> 4):
  // Input type: MX_ext<S_Prod, M_Stage1_Out> (i.e. MX_ext<S_Prod, 2*M_W+1>)
  // Output type: MX_ext<S_Prod, 2*M_W+2>
  constexpr int M_Stage2 = M_Prod + 1;
  std::vector<MX_ext<S_Prod, M_Stage2 + 1>> stage2_sums(4);
  for (size_t i = 0; i < 4; ++i) {
    stage2_sums[i] = special_add_mx_ext<S_Prod, M_Stage2>(
        stage1_sums[2 * i], stage1_sums[2 * i + 1]);
    // std::cout << "OP 1 : " << std::endl; // Debug print
    // stage1_sums[2 * i].print();
    // std::cout << "OP 2 : " << std::endl; // Debug print
    // stage1_sums[2 * i + 1].print();
    // std::cout << "Sum : " << std::endl; // Debug print
    // stage2_sums[i].print(); // Debug print
    // std::cout << std::endl;
  }

  // Stage 3 (4 -> 2):
  // Input type: MX_ext<S_Prod, M_Stage2_Out> (i.e. MX_ext<S_Prod, 2*M_W+2>)
  // Output type: MX_ext<S_Prod, 2*M_W+3>
  constexpr int M_Stage3 = M_Prod + 2;
  std::vector<MX_ext<S_Prod, M_Stage3 + 1>> stage3_sums(2);
  for (size_t i = 0; i < 2; ++i) {
    stage3_sums[i] = special_add_mx_ext<S_Prod, M_Stage3>(
        stage2_sums[2 * i], stage2_sums[2 * i + 1]);
  }

  // Stage 4 (2 -> 1): Final sum
  // Input type: MX_ext<S_Prod, M_Stage3_Out> (i.e. MX_ext<S_Prod, 2*M_W+3>)
  // Output type: MX_ext<S_Prod, 2*M_W+4>
  constexpr int M_Final = M_Prod + 3;
  MX_ext<S_Prod, M_Final + 1> final_result =
      special_add_mx_ext<S_Prod, M_Final>(stage3_sums[0], stage3_sums[1]);
  // std::cout << "OP1 : " << std::endl; // Debug print
  // stage3_sums[0].print(); // Debug print
  // std::cout << "OP2 : " << std::endl; // Debug print
  // stage3_sums[1].print(); // Debug print
  // std::cout << "Final Result: " << std::endl; // Debug print
  // final_result.print(); // Debug print

  return final_result;
}

#endif
