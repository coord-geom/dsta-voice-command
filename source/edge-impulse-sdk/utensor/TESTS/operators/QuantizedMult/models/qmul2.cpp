// Auto generated by utensor-cli

#include "qmul2_weight.hpp"
#include "qmul2.hpp"
#include "src/uTensor/ops/ArrayOps.hpp"
#include "src/uTensor/core/context.hpp"
#include "src/uTensor/ops/MathOps.hpp"
#include "src/uTensor/core/tensor.hpp"


void get_qmul2_ctx(Context& ctx) {
{    
    ctx.add(new BinaryTensor<float>({10}, inline_ref_2_0), 
            "ref_2:0");
}
{    
    ctx.add(new BinaryTensor<float>({10}, inline_a_2_0), 
            "a_2:0", 
            2);
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_c_2_eightbit_a_2__port__0_reshape_dims_0), 
            "c_2_eightbit/a_2__port__0/reshape_dims:0", 
            1);
}
{
    ctx.add(new RamTensor<float>(), "c_2_eightbit/a_2__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(), 
             { "a_2:0", "c_2_eightbit/a_2__port__0/reshape_dims:0" },
             { "c_2_eightbit/a_2__port__0/reshape:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_c_2_eightbit_a_2__port__0_reduction_dims_0), 
            "c_2_eightbit/a_2__port__0/reduction_dims:0", 
            2);
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "c_2_eightbit/a_2__port__0/min:0", 1);
    ctx.push(new MinOp(), 
             { "c_2_eightbit/a_2__port__0/reshape:0", "c_2_eightbit/a_2__port__0/reduction_dims:0" },
             { "c_2_eightbit/a_2__port__0/min:0" });
    ctx.eval();
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "c_2_eightbit/a_2__port__0/max:0", 1);
    ctx.push(new MaxOp(), 
             { "c_2_eightbit/a_2__port__0/reshape:0", "c_2_eightbit/a_2__port__0/reduction_dims:0" },
             { "c_2_eightbit/a_2__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "c_2_eightbit/a_2__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "c_2_eightbit/a_2__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "c_2_eightbit/a_2__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "a_2:0",  "c_2_eightbit/a_2__port__0/min:0", "c_2_eightbit/a_2__port__0/max:0" },
             {  "c_2_eightbit/a_2__port__0/quantize:0",  "c_2_eightbit/a_2__port__0/quantize:1", "c_2_eightbit/a_2__port__0/quantize:2" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<float>({10}, inline_b_2_0), 
            "b_2:0", 
            2);
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_c_2_eightbit_b_2__port__0_reshape_dims_0), 
            "c_2_eightbit/b_2__port__0/reshape_dims:0", 
            1);
}
{
    ctx.add(new RamTensor<float>(), "c_2_eightbit/b_2__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(), 
             { "b_2:0", "c_2_eightbit/b_2__port__0/reshape_dims:0" },
             { "c_2_eightbit/b_2__port__0/reshape:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_c_2_eightbit_b_2__port__0_reduction_dims_0), 
            "c_2_eightbit/b_2__port__0/reduction_dims:0", 
            2);
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "c_2_eightbit/b_2__port__0/min:0", 1);
    ctx.push(new MinOp(), 
             { "c_2_eightbit/b_2__port__0/reshape:0", "c_2_eightbit/b_2__port__0/reduction_dims:0" },
             { "c_2_eightbit/b_2__port__0/min:0" });
    ctx.eval();
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "c_2_eightbit/b_2__port__0/max:0", 1);
    ctx.push(new MaxOp(), 
             { "c_2_eightbit/b_2__port__0/reshape:0", "c_2_eightbit/b_2__port__0/reduction_dims:0" },
             { "c_2_eightbit/b_2__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "c_2_eightbit/b_2__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "c_2_eightbit/b_2__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "c_2_eightbit/b_2__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "b_2:0",  "c_2_eightbit/b_2__port__0/min:0", "c_2_eightbit/b_2__port__0/max:0" },
             {  "c_2_eightbit/b_2__port__0/quantize:0",  "c_2_eightbit/b_2__port__0/quantize:1", "c_2_eightbit/b_2__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "c_2/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "c_2/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "c_2/eightbit:2", 2);
    ctx.push(new QuantizedMulOp<uint8_t, uint8_t, int>(), 
             { "c_2_eightbit/a_2__port__0/quantize:0", "c_2_eightbit/a_2__port__0/quantize:1", "c_2_eightbit/a_2__port__0/quantize:2", "c_2_eightbit/b_2__port__0/quantize:0", "c_2_eightbit/b_2__port__0/quantize:1",  "c_2_eightbit/b_2__port__0/quantize:2" },
             { "c_2/eightbit:0", "c_2/eightbit:1",  "c_2/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "c_2/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "c_2/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "c_2/eightbit:0", "c_2/eightbit:1", "c_2/eightbit:2" },
             { "c_2/eightbit/requant_range:0", "c_2/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "c_2/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "c_2/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "c_2/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "c_2/eightbit:0", "c_2/eightbit:1", "c_2/eightbit:2", "c_2/eightbit/requant_range:0", "c_2/eightbit/requant_range:1" },
             { "c_2/eightbit/requantize:0", "c_2/eightbit/requantize:1", "c_2/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>(), "c_2:0");
    ctx.push(new DequantizeOp(), 
             { "c_2/eightbit/requantize:0", "c_2/eightbit/requantize:1", "c_2/eightbit/requantize:2" },
             { "c_2:0" });
}
}
