import torch

class Quantization:
    def __init__(self, 
                 in_dtype: torch.dtype=torch.float32,
                 out_dtype: torch.dtype=torch.float16):
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def quantize_weight(self, weight: torch.Tensor):
        self.scale = None
        if weight.dtype != self.in_dtype:
            raise ValueError
        if self.in_dtype == torch.float32 and self.out_dtype == torch.float16:
            quantized_weight, self.scale = self.quantize_fp32_to_fp16(weight, 5, 10)
            return quantized_weight.to(torch.float16)
        if self.in_dtype == torch.float32 and self.out_dtype == torch.bfloat16:
            quantized_weight, self.scale = self.quantize_fp32_to_fp16(weight, 8, 7)
            return quantized_weight.to(torch.bfloat16)
        if self.in_dtype == torch.float32 and self.out_dtype == torch.float8_e5m2:
            quantized_weight, self.scale = self.quantize_fp32_to_fp8(weight, 5, 2)
            return quantized_weight.to(torch.float8_e5m2)

    def dequantize_weight(self, weight: torch.Tensor):
        if weight.dtype != self.out_dtype:
            raise ValueError
        if self.in_dtype == torch.float32 and self.out_dtype == torch.float16:
            dequantized_weight = self.dequantize_fp16_to_fp32(weight.to(torch.float32), self.scale)
            return dequantized_weight
        if self.in_dtype == torch.float32 and self.out_dtype == torch.bfloat16:
            dequantized_weight = self.dequantize_fp16_to_fp32(weight.to(torch.float32), self.scale)
            return dequantized_weight
        if self.in_dtype == torch.float32 and self.out_dtype == torch.float8_e5m2:
            dequantized_weight = self.dequantize_fp8_to_fp32(weight.to(torch.float32), self.scale)
            return dequantized_weight

    def quantize_fp32_to_fp16(self, matrix, mantissa_bits, exponent_bits):
        # Number of exponent bits determines the range and bias
        exponent_max = 2 ** exponent_bits - 1
        bias = exponent_max // 2

        # Calculating the limits based on mantissa and exponent bits
        max_exponent = float(2 ** (exponent_max - bias))
        min_exponent = float(2 ** (-bias))

        # Scale matrix to be between min_exponent and max_exponent
        max_val = torch.max(torch.abs(matrix))
        scale = max_val / max_exponent

        # Normalize and scale the matrix
        normalized_matrix = matrix / scale

        # Quantize the normalized matrix
        step = min_exponent / (2 ** mantissa_bits)
        quantized_matrix = torch.round(normalized_matrix / step) * step

        # Clip to avoid overflow
        quantized_matrix = torch.clamp(quantized_matrix, -max_exponent, max_exponent)

        return quantized_matrix, scale
    
    def dequantize_fp16_to_fp32(self, quantized_matrix, scale):
        # Dequantize the matrix by multiplying with the scale used during quantization
        dequantized_matrix = quantized_matrix * scale
        return dequantized_matrix

    def quantize_fp32_to_fp8(self, matrix, mantissa_bits, exponent_bits):
        # Number of exponent bits determines the range and bias
        exponent_max = 2 ** exponent_bits - 1
        bias = exponent_max // 2

        # Calculating the limits based on mantissa and exponent bits
        max_exponent = float(2 ** (exponent_max - bias))
        min_exponent = float(2 ** (-bias))

        # Scale matrix to be between min_exponent and max_exponent
        max_val = torch.max(torch.abs(matrix))
        scale = max_val / max_exponent

        # Normalize and scale the matrix
        normalized_matrix = matrix / scale

        # Quantize the normalized matrix
        step = min_exponent / (2 ** mantissa_bits)
        quantized_matrix = torch.round(normalized_matrix / step) * step

        # Clip to avoid overflow
        quantized_matrix = torch.clamp(quantized_matrix, -max_exponent, max_exponent)

        return quantized_matrix, scale
    
    def dequantize_fp8_to_fp32(self, quantized_matrix, scale):
        # Dequantize the matrix by multiplying with the scale used during quantization
        dequantized_matrix = quantized_matrix * scale
        return dequantized_matrix
    