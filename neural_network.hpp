#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

class NeuralNetwork{

int num_inputs, num_hidden, num_output;

std::vector<double> inputs, hidden_biases, hidden_outputs, output_biases, outputs;
std::vector<std::vector<double>> input_hidden_weights, hidden_output_weights;

public:
   
   NeuralNetwork(int i, int h, int o)
   :num_inputs(i), num_hidden(h), num_output(o),
    inputs(i, 0), hidden_biases(h), hidden_outputs(h),
    input_hidden_weights(i, std::vector<double>(h,0.0)), 
    hidden_output_weights(h, std::vector<double>(o,0.0)),
    output_biases(o), outputs(o)
   {
      std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
      std::default_random_engine re;

      for(int j = 0; j < i; ++j)
        for(int m = 0; m < h; ++m)
         input_hidden_weights[j][m] = unif(re);

      for(j = 0; j < h; ++j)
         hidden_biases[j] = unif(re);

      for(j = 0; j < h; ++j)
        for(m = 0; m < o; ++m)
         hidden_output_weights[j][m] = unif(re);

      for(j = 0; j < o; ++j)
         output_biases[i] = unif(re);
   }

  std::vector<double>
  all_weights()
  {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      std::vector<double> result(numWeights);
      int k = 0;

      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j, ++k)
          result[k] = ihWeights[i][j];

      for (int i = 0; i < hBiases.Length; ++i, ++k)
        result[k] = hBiases[i];

      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j, ++k)
          result[k] = hoWeights[i][j];

      for (int i = 0; i < oBiases.Length; ++i, ++k)
        result[k] = oBiases[i];

      return result;
  }

  void
  copy_weights(std::vector<double> new_values)
  {
    int num_weights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    assert(num_weights == new_values.size());
    std::vector<double> result(numWeights);
    int k = 0;

      for (int i = 0; i < input_hidden_weights.Length; ++i)
        for (int j = 0; j < input_hidden_weights[0].Length; ++j, ++k)
          result[k] = input_hidden_weights[i][j];

      for (int i = 0; i < hidden_biases.Length; ++i, ++k)
        result[k] = hidden_biases[i];

      for (int i = 0; i < hidden_output_weights.Length; ++i)
        for (int j = 0; j < hidden_output_weights[0].Length; ++j, ++k)
          result[k] = hidden_output_weights[i][j];

      for (int i = 0; i < output_biases.Length; ++i, ++k)
        result[k] = output_biases[i];

      return result;
  }

  double
  hyper_tan(double x)
  {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return std::tanh(x);
  }

 std::vector<double>
 soft_max(std::vector<double> out_sum)
 {
   double max = out_sums[0];

   for(int i = 0; i < out_sums.size(); ++i)
     if(out_sums[i] > max) max = out_sums[i];

   double scale = 0;
   for(i = 0; i < out_sums.size(); ++i)
     scale += std::exp(out_sums[i] - max);

   std::vector<double> result(out_sums.size());
   for(i = 0; i < out_sums.size(); ++i)
     result[i] = std::exp(out_sums[i] - max) / scale;

   return result;
 } 

  int
  max_value_idx(std::vector<double> v)
  {
    int maxIdx = 0;
    double max = v[0];

    for(int i = 0; i < v.size(); ++i)
      if(v[i] > max)
       { 
         max = v[i];
         maxIdx = i;
       }

    return max;
  }

  std::vector<double>
  compute_values(std::vector<double> in_vals) 
  {
    std::vector<double> hid_sums(num_hidden, 0), out_sums(num_output, 0);

    for(int i = 0; i < num_hidden; ++i)
       for(int j = 0; j < num_inputs; ++j)
          hid_sums[i] = in_vals[j] * input_hidden_weights[j][i];

    for(i = 0; i < num_hidden; ++i)
       hid_sums[i] += hidden_biases[i];
    
    for(i = 0; i < num_hidden; ++i)
       hidden_outputs[i] = hyper_tan(hid_sums[i]);
    
    for(i = 0; i < num_output; ++i)
       for(j = 0; j < num_hidden; ++j)
          out_sums[i] += hidden_outputs[j] * hidden_output_weights[j][i];
    
    for(i = 0; i < num_output; ++i)
       out_sums[i] += output_biases[i];
    
    outputs = soft_max(out_sums);

    return outputs;
      
  }

  double
  accuracy(std::vector<std::vector<double>> test_data, std::vector<double> weights)
  {
    copy_weights(weights);

    int right = 0;
    int wrong = 0;
    std::vector<double> in_vals(num_inputs), tar_vals(num_output), comp_vals;

    for(int i = 0; i < test_data.size(); ++i)
    {
      for(int j = 0; j < num_inputs; ++j)
          in_vals[j] = data[i][j];

      for(j = 0; j < num_output; ++j)
          target_vals[j] = data[i][j];

      comp_vals = compute_values(in_vals); 
      int maxIdx = max_value_idx(comp_vals);

      if(tar_vals[maxIdx] == 1)
        ++right;
      else
        ++wrong; 
    }
    return (right * 1.0) / (right + wrong);
  }

  double 
  sq_mean_err(std::vector<std::vector<double>> data, std::vector<double> weights)
  {
     std::vector<double> input_vals(num_inputs), target_vals(num_output);
     double err = 0;

     for(int i = 0; i < data.size(); ++i)
     {
        for(int j = 0; j < num_inputs; ++j)
          input_vals[j] = data[i][j];

        for(j = 0; j < num_output; ++j)
          target_vals[j] = data[i][j];

        std::vector<double> new_values = compute_values(input_vals);

        for(i = 0; i < new_values.size(); ++i)
           err += std::pow((new_values[i] - target_vals[i]), 2);
     }

     return err / data.size();
  }

  std::vector<double>
  make_a_man_out_of_you(std::vector<std::vector<double>> t_data, int max_epochs)
  {
     std::vector<double> im_hid_grad(num_hidden);
     std::vector<double> im_out_grad(num_output);

     std::vector<std::vector<double>> all_hout_grad(num_hidden, std::vector<double>(num_output, 0));
     std::vector<std::vector<double>> all_hin_grad(num_inputs, std::vector<double>(num_hidden, 0));
     std::vector<double> out_grad(num_output);
     std::vector<double> hid_grad(num_hidden);

     std::vector<std::vector<double>> all_hout_grad_prev(num_hidden, std::vector<double>(num_output, 0.01));
     std::vector<std::vector<double>> all_hin_grad_prev(num_inputs, std::vector<double>(num_hidden, 0.01));
     std::vector<double> out_grad_prev(num_output);
     std::vector<double> hid_grad_prev(num_hidden);

     std::vector<std::vector<double>> all_hout_delta_prev(num_hidden, std::vector<double>(num_output, 0.01));
     std::vector<std::vector<double>> all_hin_delta_prev(num_inputs, std::vector<double>(num_hidden, 0.01));
     std::vector<double> out_delta_prev(num_output, 0.01);
     std::vector<double> hid_delta_prev(num_hidden, 0.01);

     double etaPlus = 1.2; // values are from the paper
     double etaMinus = 0.5;
     double deltaMax = 50.0;
     double deltaMin = 1.0E-6;

     int epoch = 0;
           while (epoch < max_epochs)
      {
        ++epoch;

        if (epoch % 100 == 0 && epoch != max_epochs)
        {
          auto currWts = all_weights();
          double err = sq_mean_err(t_data, currWts);
          std::cout << "epoch = " << epoch << " err = " << err << std::endl;
        }

        // 1. compute and accumulate all gradients
        all_hout_grad = std::vector<std::vector<double>>(num_hidden, std::vector<double>(num_output, 0)); // zero-out values from prev iteration
        all_hin_grad = std::vector<std::vector<double>>(num_inputs, std::vector<double>(num_hidden, 0));
        out_grad = std::vector<double>(num_output, 0);
        hid_grad = std::vector<double>(num_hidden, 0);

        auto xValues = std::vector<double>(numInput); // inputs
        auto tValues = std::vector<double>(numOutput); // target values
//------------------------------------------------------------------------------------------------------------------
        for (int row = 0; row < t_data.size(); ++row)  // walk thru all training data
        {
          // no need to visit in random order because all rows processed before any updates ('batch')
          Array.Copy(t_data[row], xValues, numInput); // get the inputs
          Array.Copy(t_data[row], numInput, tValues, 0, numOutput); // get the target values
          ComputeOutputs(xValues); // copy xValues in, compute outputs using curr weights (and store outputs internally)

          // compute the h-o gradient term/component as in regular back-prop
          // this term usually is lower case Greek delta but there are too many other deltas below
          for (int i = 0; i < numOutput; ++i)
          {
            double derivative = (1 - outputs[i]) * outputs[i]; // derivative of softmax = (1 - y) * y (same as log-sigmoid)
            oGradTerms[i] = derivative * (outputs[i] - tValues[i]); // careful with O-T vs. T-O, O-T is the most usual
          }

          // compute the i-h gradient term/component as in regular back-prop
          for (int i = 0; i < numHidden; ++i)
          {
            double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // derivative of tanh = (1 - y) * (1 + y)
            double sum = 0.0;
            for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
            {
              double x = oGradTerms[j] * hoWeights[i][j];
              sum += x;
            }
            hGradTerms[i] = derivative * sum;
          }

          // add input to h-o component to make h-o weight gradients, and accumulate
          for (int i = 0; i < numHidden; ++i)
          {
            for (int j = 0; j < numOutput; ++j)
            {
              double grad = oGradTerms[j] * hOutputs[i];
              all_hout_grad[i][j] += grad;
            }
          }

          // the (hidden-to-) output bias gradients
          for (int i = 0; i < numOutput; ++i)
          {
            double grad = oGradTerms[i] * 1.0; // dummy input
            out_grad[i] += grad;
          }

          // add input term to i-h component to make i-h weight gradients and accumulate
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double grad = hGradTerms[j] * inputs[i];
              all_hin_grad[i][j] += grad;
            }
          }

          // the (input-to-) hidden bias gradient
          for (int i = 0; i < numHidden; ++i)
          {
            double grad = hGradTerms[i] * 1.0;
            hid_grad[i] += grad;
          }
        } // each row
        // end compute all gradients

        // update all weights and biases (in any order)

        // update input-hidden weights
        double delta = 0.0;

        for (int i = 0; i < numInput; ++i)
        {
          for (int j = 0; j < numHidden; ++j)
          {
            if (ihPrevWeightGradsAcc[i][j] * all_hin_grad[i][j] > 0) // no sign change, increase delta
            {
              delta = ihPrevWeightDeltas[i][j] * etaPlus; // compute delta
              if (delta > deltaMax) delta = deltaMax; // keep it in range
              double tmp = -Math.Sign(all_hin_grad[i][j]) * delta; // determine direction and magnitude
              ihWeights[i][j] += tmp; // update weights
            }
            else if (ihPrevWeightGradsAcc[i][j] * all_hin_grad[i][j] < 0) // grad changed sign, decrease delta
            {
              delta = ihPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved for later)
              if (delta < deltaMin) delta = deltaMin; // keep it in range
              ihWeights[i][j] -= ihPrevWeightDeltas[i][j]; // revert to previous weight
              all_hin_grad[i][j] = 0; // forces next if-then branch, next iteration
            }
            else // this happens next iteration after 2nd branch above (just had a change in gradient)
            {
              delta = ihPrevWeightDeltas[i][j]; // no change to delta
              // no way should delta be 0 . . . 
              double tmp = -Math.Sign(all_hin_grad[i][j]) * delta; // determine direction
              ihWeights[i][j] += tmp; // update
            }
            //Console.WriteLine(ihPrevWeightGradsAcc[i][j] + " " + all_hin_grad[i][j]); Console.ReadLine();

            ihPrevWeightDeltas[i][j] = delta; // save delta
            ihPrevWeightGradsAcc[i][j] = all_hin_grad[i][j]; // save the (accumulated) gradient
          } // j
        } // i

        // update (input-to-) hidden biases
        for (int i = 0; i < numHidden; ++i)
        {
          if (hPrevBiasGradsAcc[i] * hid_grad[i] > 0) // no sign change, increase delta
          {
            delta = hPrevBiasDeltas[i] * etaPlus; // compute delta
            if (delta > deltaMax) delta = deltaMax;
            double tmp = -Math.Sign(hid_grad[i]) * delta; // determine direction
            hBiases[i] += tmp; // update
          }
          else if (hPrevBiasGradsAcc[i] * hid_grad[i] < 0) // grad changed sign, decrease delta
          {
            delta = hPrevBiasDeltas[i] * etaMinus; // the delta (not used, but saved later)
            if (delta < deltaMin) delta = deltaMin;
            hBiases[i] -= hPrevBiasDeltas[i]; // revert to previous weight
            hid_grad[i] = 0; // forces next branch, next iteration
          }
          else // this happens next iteration after 2nd branch above (just had a change in gradient)
          {
            delta = hPrevBiasDeltas[i]; // no change to delta

            if (delta > deltaMax) delta = deltaMax;
            else if (delta < deltaMin) delta = deltaMin;
            // no way should delta be 0 . . . 
            double tmp = -Math.Sign(hid_grad[i]) * delta; // determine direction
            hBiases[i] += tmp; // update
          }
          hPrevBiasDeltas[i] = delta;
          hPrevBiasGradsAcc[i] = hid_grad[i];
        }

        // update hidden-to-output weights
        for (int i = 0; i < numHidden; ++i)
        {
          for (int j = 0; j < numOutput; ++j)
          {
            if (hoPrevWeightGradsAcc[i][j] * all_hout_grad[i][j] > 0) // no sign change, increase delta
            {
              delta = hoPrevWeightDeltas[i][j] * etaPlus; // compute delta
              if (delta > deltaMax) delta = deltaMax;
              double tmp = -Math.Sign(all_hout_grad[i][j]) * delta; // determine direction
              hoWeights[i][j] += tmp; // update
            }
            else if (hoPrevWeightGradsAcc[i][j] * all_hout_grad[i][j] < 0) // grad changed sign, decrease delta
            {
              delta = hoPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved later)
              if (delta < deltaMin) delta = deltaMin;
              hoWeights[i][j] -= hoPrevWeightDeltas[i][j]; // revert to previous weight
              all_hout_grad[i][j] = 0; // forces next branch, next iteration
            }
            else // this happens next iteration after 2nd branch above (just had a change in gradient)
            {
              delta = hoPrevWeightDeltas[i][j]; // no change to delta
              // no way should delta be 0 . . . 
              double tmp = -Math.Sign(all_hout_grad[i][j]) * delta; // determine direction
              hoWeights[i][j] += tmp; // update
            }
            hoPrevWeightDeltas[i][j] = delta; // save delta
            hoPrevWeightGradsAcc[i][j] = all_hout_grad[i][j]; // save the (accumulated) gradients
          } // j
        } // i

        // update (hidden-to-) output biases
        for (int i = 0; i < numOutput; ++i)
        {
          if (oPrevBiasGradsAcc[i] * out_grad[i] > 0) // no sign change, increase delta
          {
            delta = oPrevBiasDeltas[i] * etaPlus; // compute delta
            if (delta > deltaMax) delta = deltaMax;
            double tmp = -Math.Sign(out_grad[i]) * delta; // determine direction
            oBiases[i] += tmp; // update
          }
          else if (oPrevBiasGradsAcc[i] * out_grad[i] < 0) // grad changed sign, decrease delta
          {
            delta = oPrevBiasDeltas[i] * etaMinus; // the delta (not used, but saved later)
            if (delta < deltaMin) delta = deltaMin;
            oBiases[i] -= oPrevBiasDeltas[i]; // revert to previous weight
            out_grad[i] = 0; // forces next branch, next iteration
          }
          else // this happens next iteration after 2nd branch above (just had a change in gradient)
          {
            delta = oPrevBiasDeltas[i]; // no change to delta
            // no way should delta be 0 . . . 
            double tmp = -Math.Sign(hid_grad[i]) * delta; // determine direction
            oBiases[i] += tmp; // update
          }
          oPrevBiasDeltas[i] = delta;
          oPrevBiasGradsAcc[i] = out_grad[i];
        }
      } // while

      auto wts = all_weights();
      return wts;

  }

};
