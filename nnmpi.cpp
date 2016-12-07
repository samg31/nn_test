#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mpi.h>

using matrix = std::vector<std::vector<double>>;
using vec = std::vector<double>;

constexpr int num_inputs = 7;
constexpr int num_hidden = 9;
constexpr int num_output = 3;
constexpr int max_epochs = 1000;

constexpr double lower_bound = 0.0001;
constexpr double upper_bound = 0.01;

std::uniform_real_distribution<double> unif( lower_bound, upper_bound );
std::default_random_engine re;

vec inputs( num_inputs, 0.0 );
vec outputs( num_output );
vec hidden_biases( num_hidden );
vec hidden_outputs( num_hidden );
vec output_biases( num_output );

matrix in_hid_wts( num_inputs, vec( num_hidden, 0.0 ) );
matrix hid_out_wts( num_hidden, vec( num_output, 0.0 ) );

// fill vectors/matrices with a uniform random distribution
void initialize();

// determine if a number is positive. negative, or zero
int sign( double x );

// approximate tanh to 30 decimals
double tanh_approx( double x );
double sq_mean_error( matrix& data  );
vec train( matrix& t_data, int max_epochs );
vec serialize_weights();
void set_weights(vec result);
vec soft_max( vec out_sums );
vec compute_values( vec in_vals );
int max_value_idx( vec v );

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    int num_threads, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &num_threads );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if( rank == 0 )
    {
		std::cout << "Completed." << std::endl;
		matrix input( 7018, vec( 10, 0.0 ) );

		std::ifstream reader("training.dat");
		std::string current;

		std::cout << "Reading data... ";
		for(int j = 0; j < input.size(); ++j)
		{
			for(int i = 0; i < 9; ++i)
			{
				getline(reader, current, ',');
				input[j][i] = std::stod(current);
			}
			getline(reader, current);
			input[j][9] = std::stod(current);
		}

		std::cout << "Completed." << std::endl;

		std::cout << "Partitioning data table... ";
		std::size_t const partition = input.size() / (num_threads - 1);
		std::vector<matrix> partitioned_input;
		partitioned_input.reserve( num_threads - 1 );
		int j = 0;
		for( int i = 0; i < num_threads - 1; ++i )
		{
			matrix temp;
			for( ; j < partition*(i+1); ++j )
			{
				temp.push_back( input[j] );
			}
			partitioned_input.push_back( temp );
		}
		auto start = std::chrono::steady_clock::now();
		std::cout << "Completed." << std::endl;
		std::cout << "Sending partitions to slaves... ";
		for( int i = 0; i < partitioned_input.size(); ++i )
		{
			int size = partitioned_input[i].size();
			MPI_Send( &size, 1, MPI_INT, (i+1), 1, MPI_COMM_WORLD );
			for( auto& vector  : partitioned_input[i] )
				MPI_Send( &vector.front(), vector.size(), MPI_DOUBLE, (i+1), 0, MPI_COMM_WORLD );
		}
		std::cout << "Completed." << std::endl;
	
		// auto weights = train(input, max_epochs);

		// for(auto v : weights)
		//     std::cout << v << std::endl;
		int lowest = 1;
		vec errs(num_threads, 0);

		for(int i = 1; i < num_threads; ++i)
			MPI_Recv( &errs[i], 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

		for(int j = 2; j < num_threads; ++j)
			if(errs[j] < errs[lowest])
				lowest = j;

		int chosen = 0;

		for(int j = 1; j < num_threads; ++j)
			if(j != lowest)
				MPI_Send( &chosen, 1, MPI_INT, j, 2, MPI_COMM_WORLD );

		++chosen;
		MPI_Send( &chosen, 1, MPI_INT, lowest, 2, MPI_COMM_WORLD );

		vec result((num_inputs * num_hidden)
		+ (num_hidden * num_output) + num_hidden + num_output);

		MPI_Recv( &result.front(), result.size(), MPI_DOUBLE, lowest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		auto end = std::chrono::steady_clock::now();   

		std::cout << "Time: " << (end - start).count() << std::endl;

	    set_weights(result);

		vec res = compute_values(input[2063]);
		for(auto v : res)
		std::cout << v << std::endl;

    }

    else
    {
		std::cout << "Initializing values... ";
		initialize();
		int chosen = 0;
		int size;
		MPI_Recv( &size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		matrix table;
		vec row( 10 );
		table.reserve(size);
		for( int i = 0; i < size; ++i )
		{
			MPI_Recv( &row.front(), row.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
			table.push_back( row );
		}
		std::cout << "Received" <<std::endl;
		vec weights = train(table, max_epochs);

		double err = sq_mean_error(table);
		MPI_Send(&err, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
std::cout << "sent" << std::endl;
		MPI_Recv( &chosen, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
std::cout << "received" << std::endl;
		if(chosen == 1)
			MPI_Send( &weights.front(), weights.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );

    }
    MPI_Finalize();
    return 0;
}

void initialize()
{
    for( int i = 0; i < num_output; ++i )
    	output_biases[i] = unif(re);

    for( int i = 0; i < num_hidden; ++i )
	hidden_biases[i] = unif(re);

    for( int i = 0; i < num_hidden; ++i )
	for( int j = 0; j < num_output; ++j )
	    hid_out_wts[i][j] = unif(re);

    for( int i = 0; i < num_inputs; ++i )
	for( int j = 0; j < num_hidden; ++j )
	    in_hid_wts[i][j] = unif(re);
}

int sign( double x )
{
    if(x > 0)
	return 1;
    else if (x < 0)
	return -1;
    else
	return 0;    
}

double tanh_approx( double x )
{
    if (x < -20.0) return -1.0;
    else if (x > 20.0) return 1.0;
    else return std::tanh(x);
}

double sq_mean_error( matrix& data  )
{
    vec input_vals( num_inputs ), target_vals( num_output );
    double err = 0.0;

    for (int i = 0; i < data.size(); ++i)
    {
	for (int j = 0; j < num_inputs; ++j)
	    input_vals[j] = data[i][j];
            
	for (int j = num_inputs; j-num_inputs < num_output; ++j)
	    target_vals[j-num_inputs] = data[i][j];
            
	vec new_values = compute_values(input_vals);
            
	for (int q = 0; q < new_values.size(); ++q)
	    err += (new_values[q] - target_vals[q]) * (new_values[q] - target_vals[q]);
    }
        
    return err / data.size();
}
vec compute_values( vec in_vals )
{
    vec hidden_sums( num_hidden, 0 ), out_sums( num_output, 0 );
        
    for (int i = 0; i < num_hidden; ++i)
	for (int j = 0; j < num_inputs; ++j)
	    hidden_sums[i] += in_vals[j] * in_hid_wts[j][i];
        
    for (int i = 0; i < num_hidden; ++i)
	hidden_sums[i] += hidden_biases[i];
        
    for (int i = 0; i < num_hidden; ++i)
	hidden_outputs[i] = tanh_approx(hidden_sums[i]);
        
    for (int i = 0; i < num_output; ++i)
	for (int j = 0; j < num_hidden; ++j)
	    out_sums[i] += hidden_outputs[j] * hid_out_wts[j][i];
        
    for (int i = 0; i < num_output; ++i)
	out_sums[i] += output_biases[i];
        
    outputs = soft_max(out_sums);
        
    return outputs;

}

vec soft_max( vec out_sums )
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

vec serialize_weights()
{
    int num_weights = (num_inputs * num_hidden)
	+ (num_hidden * num_output) + num_hidden + num_output;
    vec result( num_weights );
    int offset = ( num_inputs * num_hidden );

    for (int i = 0; i < in_hid_wts.size(); ++i)
	for (int j = 0; j < in_hid_wts[i].size(); ++j)
	    result[(i * in_hid_wts.size()) + j] = in_hid_wts[i][j];
    for (int i = 0; i < hidden_biases.size(); ++i)
	result[(i * hidden_biases.size())+offset] = hidden_biases[i];
            
    offset += num_hidden;
            
    for (int i = 0; i < hid_out_wts.size(); ++i)
	for (int j = 0; j < hid_out_wts[0].size(); ++j)
	    result[(i * hid_out_wts.size()) + j + offset] = hid_out_wts[i][j];
            
    offset += (num_hidden * num_output);
            
    for (int i = 0; i < output_biases.size(); ++i)
	result[(i * output_biases.size()) + offset] = output_biases[i];

    return result;

}

void set_weights(vec result)
{

    int offset = ( num_inputs * num_hidden );

    for (int i = 0; i < in_hid_wts.size(); ++i)
	for (int j = 0; j < in_hid_wts[i].size(); ++j)
	    in_hid_wts[i][j] = result[(i * in_hid_wts.size()) + j];
    for (int i = 0; i < hidden_biases.size(); ++i)
	    hidden_biases[i] = result[(i * hidden_biases.size())+offset];
            
    offset += num_hidden;
            
    for (int i = 0; i < hid_out_wts.size(); ++i)
	for (int j = 0; j < hid_out_wts[0].size(); ++j)
	   hid_out_wts[i][j] = result[(i * hid_out_wts.size()) + j + offset];
            
    offset += (num_hidden * num_output);
            
    for (int i = 0; i < output_biases.size(); ++i)
	   output_biases[i] = result[(i * output_biases.size()) + offset];

}

int max_value_idx( vec v )
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

vec train( matrix& t_data, int max_epochs )
{
        std::vector<double> hGradTerms(num_hidden);
        vec oGradTerms(num_output);
        
        matrix acc_hid_out_grad(num_hidden, vec(num_output, 0));
        matrix acc_in_hid_grad(num_inputs, vec(num_hidden, 0));
        vec out_bias_grads(num_output);
        vec hid_bias_grads(num_hidden);
        
        matrix prev_acc_hid_out_grad(num_hidden, vec(num_output, 0.01));
        matrix prev_acc_in_hid_grad(num_inputs, vec(num_hidden, 0.01));
        vec prev_out_bias_grads(num_output);
        vec prev_hid_bias_grads(num_hidden);
        
        matrix hid_out_wts_deltas(num_hidden, vec(num_output, 0.01));
        matrix in_hid_wts_deltas(num_inputs, vec(num_hidden, 0.01));
        vec out_bias_deltas(num_output, 0.01);
        vec hid_bias_deltas(num_hidden, 0.01);
        
        double increment_amount = 1.2; // values are from the paper
        double decrement_amount = 0.5;
        double max_delta = 50.0;
        double min_delta = 0.000001;
        
        int epoch = 0;
        while (epoch < max_epochs)
        {
            ++epoch;
            
            if (epoch % 100 == 0 || epoch == 1 )
            {
                double err = sq_mean_error(t_data);
                std::cout << "epoch = " << epoch << " err = " << err << std::endl;
            }
            
            acc_hid_out_grad = matrix(num_hidden, vec(num_output, 0)); 
            acc_in_hid_grad = matrix(num_inputs, vec(num_hidden, 0));
            out_bias_grads = vec(num_output, 0);
            hid_bias_grads = vec(num_hidden, 0);
            
            auto xValues = vec(num_inputs); 
            auto tValues = vec(num_output); 
            
            for (int row = 0; row < t_data.size(); ++row) 
            {
                for (int i = 0; i < num_inputs; ++i)
                    xValues[i] = t_data[row][i];
                
                for (int i = num_inputs; (i - num_inputs) < num_output; ++i)
                    tValues[i-num_inputs] = t_data[row][i];
                compute_values(xValues); 
                
                
                for (int i = 0; i < num_output; ++i)
                {
                    double derivative = (1 - outputs[i]) * outputs[i]; 
                    oGradTerms[i] = derivative * (outputs[i] - tValues[i]); 
                }
                
                
                for (int i = 0; i < num_hidden; ++i)
                {
                    double derivative = (1 - hidden_outputs[i]) * (1 + hidden_outputs[i]); 
                    double sum = 0.0;
                    for (int j = 0; j < num_output; ++j) 
                    {
                        double x = oGradTerms[j] * hid_out_wts[i][j];
                        sum += x;
                    }
                    hGradTerms[i] = derivative * sum;
                }
                
                
                for (int i = 0; i < num_hidden; ++i)
                {
                    for (int j = 0; j < num_output; ++j)
                    {
                        double grad = oGradTerms[j] * hidden_outputs[i];
                        acc_hid_out_grad[i][j] += grad;
                    }
                }
                
                
                for (int i = 0; i < num_output; ++i)
                {
                    out_bias_grads[i] += oGradTerms[i];
                }
                
                
                for (int i = 0; i < num_inputs; ++i)
                {
                    for (int j = 0; j < num_hidden; ++j)
                    {
                        double grad = hGradTerms[j] * xValues[i];
                        acc_in_hid_grad[i][j] += grad;
                    }
                }
                
                
                for (int i = 0; i < num_hidden; ++i)
                {
                    hid_bias_grads[i] += hGradTerms[i];
                }
            } 

            double delta = 0.0;
            
            for (int i = 0; i < num_inputs; ++i)
            {
                for (int j = 0; j < num_hidden; ++j)
                {
                    if (prev_acc_in_hid_grad[i][j] * acc_in_hid_grad[i][j] > 0) 
                    {
                        delta = in_hid_wts_deltas[i][j] * increment_amount; 
                        if (delta > max_delta) delta = max_delta; 
                        double tmp = -sign(acc_in_hid_grad[i][j]) * delta;  and magnitude
                        in_hid_wts[i][j] += tmp;
                    }
                    else if (prev_acc_in_hid_grad[i][j] * acc_in_hid_grad[i][j] < 0) 
                    {
                        delta = in_hid_wts_deltas[i][j] * decrement_amount; 
                        if (delta < min_delta) delta = min_delta; 
                        in_hid_wts[i][j] -= in_hid_wts_deltas[i][j]; 
                        acc_in_hid_grad[i][j] = 0; 
                    }
                    else 
                    {
                        delta = in_hid_wts_deltas[i][j]; 
                        double tmp = -sign(acc_in_hid_grad[i][j]) * delta; 
                        in_hid_wts[i][j] += tmp; 
                    }
                    
                    in_hid_wts_deltas[i][j] = delta; 
                    prev_acc_in_hid_grad[i][j] = acc_in_hid_grad[i][j]; 
                } 
            } 
            
            
            for (int i = 0; i < num_hidden; ++i)
            {
                if (prev_hid_bias_grads[i] * hid_bias_grads[i] > 0) 
                {
                    delta = hid_bias_deltas[i] * increment_amount; 
                    if (delta > max_delta) delta = max_delta;
                    double tmp = -sign(hid_bias_grads[i]) * delta; 
                    hidden_biases[i] += tmp;
                }
                else if (prev_hid_bias_grads[i] * hid_bias_grads[i] < 0) 
                {
                    delta = hid_bias_deltas[i] * decrement_amount; 
                    if (delta < min_delta) delta = min_delta;
                    hidden_biases[i] -= hid_bias_deltas[i]; 
                    hid_bias_grads[i] = 0; 
                }
                else 
                {
                    delta = hid_bias_deltas[i]; 
                    if (delta > max_delta) delta = max_delta;
                    else if (delta < min_delta) delta = min_delta;
         
                    double tmp = -sign(hid_bias_grads[i]) * delta; 
                    hidden_biases[i] += tmp; 
                }
                hid_bias_deltas[i] = delta;
                prev_hid_bias_grads[i] = hid_bias_grads[i];
            }
            
            
            for (int i = 0; i < num_hidden; ++i)
            {
                for (int j = 0; j < num_output; ++j)
                {
                    if (prev_acc_hid_out_grad[i][j] * acc_hid_out_grad[i][j] > 0) 
                    {
                        delta = hid_out_wts_deltas[i][j] * increment_amount; 
                        if (delta > max_delta) delta = max_delta;
                        double tmp = -sign(acc_hid_out_grad[i][j]) * delta; 
                        hid_out_wts[i][j] += tmp; 
                    }
                    else if (prev_acc_hid_out_grad[i][j] * acc_hid_out_grad[i][j] < 0) 
                    {
                        delta = hid_out_wts_deltas[i][j] * decrement_amount; 
                        if (delta < min_delta) delta = min_delta;
                        hid_out_wts[i][j] -= hid_out_wts_deltas[i][j]; 
                        acc_hid_out_grad[i][j] = 0; 
                    }
                    else 
                    {
                        delta = hid_out_wts_deltas[i][j]; 
                        double tmp = -sign(acc_hid_out_grad[i][j]) * delta; 
                        hid_out_wts[i][j] += tmp; 
                    }
                    hid_out_wts_deltas[i][j] = delta;
                    prev_acc_hid_out_grad[i][j] = acc_hid_out_grad[i][j]; 
                } 
            } 
            

            for (int i = 0; i < num_output; ++i)
            {
                if (prev_out_bias_grads[i] * out_bias_grads[i] > 0)
                {
                    delta = out_bias_deltas[i] * increment_amount; 
                    if (delta > max_delta) delta = max_delta;
                    double tmp = -sign(out_bias_grads[i]) * delta; 
                    output_biases[i] += tmp; 
                }
                else if (prev_out_bias_grads[i] * out_bias_grads[i] < 0) 
                {
                    delta = out_bias_deltas[i] * decrement_amount; 
                    if (delta < min_delta) delta = min_delta;
                    output_biases[i] -= out_bias_deltas[i]; 
                    out_bias_grads[i] = 0; 
                }
                else 
                {
                    delta = out_bias_deltas[i]; 
                    double tmp = -sign(hid_bias_grads[i]) * delta; 
                    output_biases[i] += tmp; 
                }
                out_bias_deltas[i] = delta;
                prev_out_bias_grads[i] = out_bias_grads[i];
            }
        } 
        
        auto wts = serialize_weights();
        return wts;
}
