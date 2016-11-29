#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

class NeuralNetwork {

	int num_inputs, num_hidden, num_output;

	std::vector<double> inputs, hidden_biases, hidden_outputs, output_biases, outputs;
	std::vector<std::vector<double>> input_hidden_weights, hidden_output_weights;

public:

	NeuralNetwork(int i, int h, int o)
		:num_inputs(i), num_hidden(h), num_output(o),
		inputs(i, 0), hidden_biases(h), hidden_outputs(h),
		input_hidden_weights(i, std::vector<double>(h, 0.0)),
		hidden_output_weights(h, std::vector<double>(o, 0.0)),
		output_biases(o), outputs(o)
	{
		double lower_bound = 0.0001, upper_bound = 0.001;
		std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
		std::default_random_engine re;

		for (int j = 0; j < i; ++j)
			for (int m = 0; m < h; ++m)
				input_hidden_weights[j][m] = unif(re);

		for (int j = 0; j < h; ++j)
			hidden_biases[j] = unif(re);

		for (int j = 0; j < h; ++j)
			for (int m = 0; m < o; ++m)
				hidden_output_weights[j][m] = unif(re);

		for (int j = 0; j < o; ++j)
			output_biases[i] = unif(re);
	}

	std::vector<double>
		all_weights()
	{
		int numWeights = (num_inputs * num_hidden) + (num_hidden * num_output) + num_hidden + num_output;
		std::vector<double> result(numWeights);
		int k = 0;

		for (int i = 0; i < input_hidden_weights.size(); ++i)
			for (int j = 0; j < input_hidden_weights[0].size(); ++j, ++k)
				result[k] = input_hidden_weights[i][j];

		for (int i = 0; i < hidden_biases.size(); ++i, ++k)
			result[k] = hidden_biases[i];

		for (int i = 0; i < hidden_output_weights.size(); ++i)
			for (int j = 0; j < hidden_output_weights[0].size(); ++j, ++k)
				result[k] = hidden_output_weights[i][j];

		for (int i = 0; i < output_biases.size(); ++i, ++k)
			result[k] = output_biases[i];

		return result;
	}

	void
		copy_weights(std::vector<double> new_values)
	{
		int numWeights = (num_inputs * num_hidden) + (num_hidden * num_output) + num_hidden + num_output;
		assert(numWeights == new_values.size());
		std::vector<double> result(numWeights);
		int k = 0;

		for (int i = 0; i < input_hidden_weights.size(); ++i)
			for (int j = 0; j < input_hidden_weights[0].size(); ++j, ++k)
				result[k] = input_hidden_weights[i][j];

		for (int i = 0; i < hidden_biases.size(); ++i, ++k)
			result[k] = hidden_biases[i];

		for (int i = 0; i < hidden_output_weights.size(); ++i)
			for (int j = 0; j < hidden_output_weights[0].size(); ++j, ++k)
				result[k] = hidden_output_weights[i][j];

		for (int i = 0; i < output_biases.size(); ++i, ++k)
			result[k] = output_biases[i];
	}

	double
		hyper_tan(double x)
	{
		if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
		else if (x > 20.0) return 1.0;
		else return std::tanh(x);
	}

	std::vector<double>
		soft_max(std::vector<double> out_sums)
	{
		double max = out_sums[0];

		for (int i = 0; i < out_sums.size(); ++i)
			if (out_sums[i] > max) max = out_sums[i];

		double scale = 0;
		for (int i = 0; i < out_sums.size(); ++i)
			scale += std::exp(out_sums[i] - max);

		std::vector<double> result(out_sums.size());
		for (int i = 0; i < out_sums.size(); ++i)
			result[i] = std::exp(out_sums[i] - max) / scale;

		return result;
	}

	int
		max_value_idx(std::vector<double> v)
	{
		int maxIdx = 0;
		double max = v[0];

		for (int i = 1; i < v.size(); ++i)
			if (v[i] > max)
			{
				max = v[i];
				maxIdx = i;
			}

		return maxIdx;
	}

	std::vector<double>
		compute_values(std::vector<double> in_vals)
	{
		std::vector<double> hid_sums(num_hidden, 0), out_sums(num_output, 0);

		for (int i = 0; i < num_hidden; ++i)
			for (int j = 0; j < num_inputs; ++j)
				hid_sums[i] += in_vals[j] * input_hidden_weights[j][i];

		for (int i = 0; i < num_hidden; ++i)
			hid_sums[i] += hidden_biases[i];

		for (int i = 0; i < num_hidden; ++i)
			hidden_outputs[i] = hyper_tan(hid_sums[i]);

		for (int i = 0; i < num_output; ++i)
			for (int j = 0; j < num_hidden; ++j)
				out_sums[i] += hidden_outputs[j] * hidden_output_weights[j][i];

		for (int i = 0; i < num_output; ++i)
			out_sums[i] += output_biases[i];

		outputs = soft_max(out_sums);

		return outputs;

	}

	double
		accuracy(std::vector<std::vector<double>> &test_data, std::vector<double> weights)
	{
		copy_weights(weights);

		int right = 0;
		int wrong = 0;
		std::vector<double> in_vals(num_inputs), tar_vals(num_output), comp_vals;

		for (int i = 0; i < test_data.size(); ++i)
		{
			for (int j = 0; j < num_inputs; ++j)
				in_vals[j] = test_data[i][j];

			for (int j = 0; j < num_output; ++j)
				tar_vals[j] = test_data[i][j];

			comp_vals = compute_values(in_vals);
			int maxIdx = max_value_idx(comp_vals);

			if (tar_vals[maxIdx] == 1)
				++right;
			else
				++wrong;
		}
		return (right * 1.0) / (right + wrong);
	}

	double
		sq_mean_err(std::vector<std::vector<double>> &data, std::vector<double> weights)
	{
		std::vector<double> input_vals(num_inputs), target_vals(num_output);
		double err = 0;

		for (int i = 0; i < data.size(); ++i)
		{
			for (int j = 0; j < num_inputs; ++j)
				input_vals[j] = data[i][j];

			for (int j = 0; j < num_output; ++j)
				target_vals[j] = data[i][j];

			std::vector<double> new_values = compute_values(input_vals);

			for (i = 0; i < new_values.size(); ++i)
				err += std::pow((new_values[i] - target_vals[i]), 2);
		}

		return err / data.size();
	}

	std::vector<double>
		make_a_man_out_of_you(std::vector<std::vector<double>> &t_data, int max_epochs)
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

			auto xValues = std::vector<double>(num_inputs); // inputs
			auto tValues = std::vector<double>(num_output); // target values

			for (int row = 0; row < t_data.size(); ++row)  // walk thru all training data
			{
				for (int i = 0; i < num_inputs; ++i)
					xValues[i] = t_data[row][i];
				for (int i = num_inputs; i < num_output; ++i)
					tValues[i] = t_data[row][i];
				compute_values(xValues); // copy xValues in, compute outputs using curr weights (and store outputs internally)

										 // compute the h-o gradient term/component as in regular back-prop
										 // this term usually is lower case Greek delta but there are too many other deltas below
				for (int i = 0; i < num_output; ++i)
				{
					double derivative = (1 - outputs[i]) * outputs[i]; // derivative of softmax = (1 - y) * y (same as log-sigmoid)
					im_out_grad[i] = derivative * (outputs[i] - tValues[i]); // careful with O-T vs. T-O, O-T is the most usual
				}

				// compute the i-h gradient term/component as in regular back-prop
				for (int i = 0; i < num_hidden; ++i)
				{
					double derivative = (1 - hidden_outputs[i]) * (1 + hidden_outputs[i]); // derivative of tanh = (1 - y) * (1 + y)
					double sum = 0.0;
					for (int j = 0; j < num_output; ++j) // each hidden delta is the sum of num_output terms
					{
						double x = im_out_grad[j] * hidden_output_weights[i][j];
						sum += x;
					}
					im_hid_grad[i] = derivative * sum;
				}

				// add input to h-o component to make h-o weight gradients, and accumulate
				for (int i = 0; i < num_hidden; ++i)
				{
					for (int j = 0; j < num_output; ++j)
					{
						double grad = im_out_grad[j] * hidden_outputs[i];
						all_hout_grad[i][j] += grad;
					}
				}

				// the (hidden-to-) output bias gradients
				for (int i = 0; i < num_output; ++i)
				{
					double grad = im_out_grad[i] * 1.0; // dummy input
					out_grad[i] += grad;
				}

				// add input term to i-h component to make i-h weight gradients and accumulate
				for (int i = 0; i < num_inputs; ++i)
				{
					for (int j = 0; j < num_hidden; ++j)
					{
						double grad = im_hid_grad[j] * inputs[i];
						all_hin_grad[i][j] += grad;
					}
				}

				// the (input-to-) hidden bias gradient
				for (int i = 0; i < num_hidden; ++i)
				{
					double grad = im_hid_grad[i] * 1.0;
					hid_grad[i] += grad;
				}
			} // each row
			  // end compute all gradients

			  // update all weights and biases (in any order)

			  // update input-hidden weights
			double delta = 0.0;

			for (int i = 0; i < num_inputs; ++i)
			{
				for (int j = 0; j < num_hidden; ++j)
				{
					if (all_hin_grad_prev[i][j] * all_hin_grad[i][j] > 0) // no sign change, increase delta
					{
						delta = all_hin_delta_prev[i][j] * etaPlus; // compute delta
						if (delta > deltaMax) delta = deltaMax; // keep it in range
						double tmp = -(all_hin_grad[i][j] / std::abs(all_hin_grad[i][j])) * delta; // determine direction and magnitude
						input_hidden_weights[i][j] += tmp; // update weights
					}
					else if (all_hin_grad_prev[i][j] * all_hin_grad[i][j] < 0) // grad changed sign, decrease delta
					{
						delta = all_hin_delta_prev[i][j] * etaMinus; // the delta (not used, but saved for later)
						if (delta < deltaMin) delta = deltaMin; // keep it in range
						input_hidden_weights[i][j] -= all_hin_delta_prev[i][j]; // revert to previous weight
						all_hin_grad[i][j] = 0; // forces next if-then branch, next iteration
					}
					else // this happens next iteration after 2nd branch above (just had a change in gradient)
					{
						delta = all_hin_delta_prev[i][j]; // no change to delta
														  // no way should delta be 0 . . . 
						double tmp = -(all_hin_grad[i][j] / std::abs(all_hin_grad[i][j])) * delta; // determine direction
						input_hidden_weights[i][j] += tmp; // update
					}
					//Console.WriteLine(all_hin_grad_prev[i][j] + " " + all_hin_grad[i][j]); Console.ReadLine();

					all_hin_delta_prev[i][j] = delta; // save delta
					all_hin_grad_prev[i][j] = all_hin_grad[i][j]; // save the (accumulated) gradient
				} // j
			} // i

			  // update (input-to-) hidden biases
			for (int i = 0; i < num_hidden; ++i)
			{
				if (hid_grad_prev[i] * hid_grad[i] > 0) // no sign change, increase delta
				{
					delta = hid_delta_prev[i] * etaPlus; // compute delta
					if (delta > deltaMax) delta = deltaMax;
					double tmp = -(hid_grad[i] / std::abs(hid_grad[i])) * delta; // determine direction
					hidden_biases[i] += tmp; // update
				}
				else if (hid_grad_prev[i] * hid_grad[i] < 0) // grad changed sign, decrease delta
				{
					delta = hid_delta_prev[i] * etaMinus; // the delta (not used, but saved later)
					if (delta < deltaMin) delta = deltaMin;
					hidden_biases[i] -= hid_delta_prev[i]; // revert to previous weight
					hid_grad[i] = 0; // forces next branch, next iteration
				}
				else // this happens next iteration after 2nd branch above (just had a change in gradient)
				{
					delta = hid_delta_prev[i]; // no change to delta

					if (delta > deltaMax) delta = deltaMax;
					else if (delta < deltaMin) delta = deltaMin;
					// no way should delta be 0 . . . 
					double tmp = -(hid_grad[i] / std::abs(hid_grad[i])) * delta; // determine direction
					hidden_biases[i] += tmp; // update
				}
				hid_delta_prev[i] = delta;
				hid_grad_prev[i] = hid_grad[i];
			}

			// update hidden-to-output weights
			for (int i = 0; i < num_hidden; ++i)
			{
				for (int j = 0; j < num_output; ++j)
				{
					if (all_hout_grad_prev[i][j] * all_hout_grad[i][j] > 0) // no sign change, increase delta
					{
						delta = all_hout_delta_prev[i][j] * etaPlus; // compute delta
						if (delta > deltaMax) delta = deltaMax;
						double tmp = -(all_hout_grad[i][j] / std::abs(all_hout_grad[i][j])) * delta; // determine direction
						hidden_output_weights[i][j] += tmp; // update
					}
					else if (all_hout_grad_prev[i][j] * all_hout_grad[i][j] < 0) // grad changed sign, decrease delta
					{
						delta = all_hout_delta_prev[i][j] * etaMinus; // the delta (not used, but saved later)
						if (delta < deltaMin) delta = deltaMin;
						hidden_output_weights[i][j] -= all_hout_delta_prev[i][j]; // revert to previous weight
						all_hout_grad[i][j] = 0; // forces next branch, next iteration
					}
					else // this happens next iteration after 2nd branch above (just had a change in gradient)
					{
						delta = all_hout_delta_prev[i][j]; // no change to delta
														   // no way should delta be 0 . . . 
						double tmp = -(all_hout_grad[i][j] / std::abs(all_hout_grad[i][j])) * delta; // determine direction
						hidden_output_weights[i][j] += tmp; // update
					}
					all_hout_delta_prev[i][j] = delta; // save delta
					all_hout_grad_prev[i][j] = all_hout_grad[i][j]; // save the (accumulated) gradients
				} // j
			} // i

			  // update (hidden-to-) output biases
			for (int i = 0; i < num_output; ++i)
			{
				if (out_grad_prev[i] * out_grad[i] > 0) // no sign change, increase delta
				{
					delta = out_delta_prev[i] * etaPlus; // compute delta
					if (delta > deltaMax) delta = deltaMax;
					double tmp = -(out_grad[i] / std::abs(out_grad[i])) * delta; // determine direction
					output_biases[i] += tmp; // update
				}
				else if (out_grad_prev[i] * out_grad[i] < 0) // grad changed sign, decrease delta
				{
					delta = out_delta_prev[i] * etaMinus; // the delta (not used, but saved later)
					if (delta < deltaMin) delta = deltaMin;
					output_biases[i] -= out_delta_prev[i]; // revert to previous weight
					out_grad[i] = 0; // forces next branch, next iteration
				}
				else // this happens next iteration after 2nd branch above (just had a change in gradient)
				{
					delta = out_delta_prev[i]; // no change to delta
											   // no way should delta be 0 . . . 
					double tmp = -(hid_grad[i] / std::abs(hid_grad[i])) * delta; // determine direction
					output_biases[i] += tmp; // update
				}
				out_delta_prev[i] = delta;
				out_grad_prev[i] = out_grad[i];
			}
		} // while

		auto wts = all_weights();
		return wts;

	}

};

#endif
