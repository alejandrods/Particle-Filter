/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

//static default_random_engine gen;

//----------------------------------------------------------------------------------
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    //Number of particles:
    num_particles = 100;
    
    //Random engine: gen
    std::default_random_engine gen;
    
    //Sensor noise
    std::normal_distribution<double> N_x(x, std[0]);
    std::normal_distribution<double> N_y(y, std[1]);
    std::normal_distribution<double> N_theta(0, std[2]);
    
    //Initialize part:
    int i;
    for (i = 0; i < num_particles; i++){
        Particle particle;
        
        particle.id = i;
        
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight = 1.0;
        
        particles.push_back(particle);
        weights.push_back(1);
    }
    
    is_initialized = true;
}

//----------------------------------------------------------------------------------

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    //Random engine: gen
    default_random_engine gen;
    
    int i;
    for (i = 0; i < num_particles; i++)
    {
        //New variables
        double new_x;
        double new_y;
        double new_theta;
        
        //Condition for the yaw_rate
        //fabs(yaw_rate) to fix the problem with "yaw error larger than the max"
        if (fabs(yaw_rate) < 1e-10) {
            new_theta = particles[i].theta;
            new_x = particles[i].x + (velocity * delta_t * cos(new_theta));
            new_y = particles[i].y + (velocity * delta_t * sin(new_theta));
            
        }
        else
        {
            new_theta = particles[i].theta + yaw_rate * delta_t;
            new_x = particles[i].x + (velocity / yaw_rate) * (sin(new_theta) - sin(particles[i].theta));
            new_y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(new_theta));
        }
        
        // Creates normal distribution for each parameter: x, y and theta
        normal_distribution<double> N_x(new_x, std_pos[0]);
        normal_distribution<double> N_y(new_y, std_pos[1]);
        normal_distribution<double> N_theta(new_theta, std_pos[2]);
        
        //Evaluate
        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);
    }
}


//----------------------------------------------------------------------------------


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    //Define a error value, for instance 1.002e99
    const double error = 1.002e99;
    
    int i,j,current_j;
    //For each particle
    for (int i = 0; i < observations.size(); i++) {
        
        double current_smallest_error = error;
        
        for (int j = 0; j < predicted.size(); j++) {
            double pred_x = predicted[j].x;
            double pred_y = predicted[j].y;
            
            //Define pred_x, pred_y and dif with obs_x and obs_y
            const double dx = pred_x - observations[i].x;
            const double dy = pred_y - observations[i].y;
            
            //Calculate error
            const double error = dx * dx + dy * dy;
            
            //Condition for the calculated error
            if (error < current_smallest_error) {
                current_j = j;
                current_smallest_error = error;
            }
        }
        observations[i].id = current_j;
    }
}

//----------------------------------------------------------------------------------


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    //Source: https://github.com/mvirgo/Kidnapped-Vehicle-Project/blob/master/src/particle_filter.cpp
    
    // First term of multi-variate normal Gaussian distribution calculated below
    // It stays the same so can be outside the loop
    const double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    
    // The denominators of the mvGd also stay the same
    const double x_denom = 2 * std_landmark[0] * std_landmark[0];
    const double y_denom = 2 * std_landmark[1] * std_landmark[1];
    
    // Iterate through each particle
    for (int i = 0; i < num_particles; ++i) {
        
        // For calculating multi-variate Gaussian distribution of each observation, for each particle
        double mvGd = 1.0;
        
        // For each observation
        for (int j = 0; j < observations.size(); ++j) {
            
            // Transform the observation point (from vehicle coordinates to map coordinates)
            double trans_obs_x, trans_obs_y;
            trans_obs_x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
            trans_obs_y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
            
            // Find nearest landmark
            vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
            vector<double> landmark_obs_dist (landmarks.size());
            for (int k = 0; k < landmarks.size(); ++k) {
                
                // Down-size possible amount of landmarks to look at by only looking at those in sensor range of the particle
                // If in range, put in the distance vector for calculating nearest neighbor
                double landmark_part_dist = sqrt(pow(particles[i].x - landmarks[k].x_f, 2) + pow(particles[i].y - landmarks[k].y_f, 2));
                if (landmark_part_dist <= sensor_range) {
                    landmark_obs_dist[k] = sqrt(pow(trans_obs_x - landmarks[k].x_f, 2) + pow(trans_obs_y - landmarks[k].y_f, 2));
                    
                } else {
                    // Need to fill those outside of distance with huge number, or they'll be a zero (and think they are closest)
                    landmark_obs_dist[k] = 999999.0;
                    
                }
                
            }
            
            // Associate the observation point with its nearest landmark neighbor
            int min_pos = distance(landmark_obs_dist.begin(),min_element(landmark_obs_dist.begin(),landmark_obs_dist.end()));
            float nn_x = landmarks[min_pos].x_f;
            float nn_y = landmarks[min_pos].y_f;
            
            // Calculate multi-variate Gaussian distribution
            double x_diff = trans_obs_x - nn_x;
            double y_diff = trans_obs_y - nn_y;
            double b = ((x_diff * x_diff) / x_denom) + ((y_diff * y_diff) / y_denom);
            mvGd *= a * exp(-b);
            
        }
        
        // Update particle weights with combined multi-variate Gaussian distribution
        particles[i].weight = mvGd;
        weights[i] = particles[i].weight;
        
    }
    
}

//----------------------------------------------------------------------------------


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    //Random engine: gen
    default_random_engine gen;
    
    //Discrete distribution
    discrete_distribution<int> distribution(weights.begin(),weights.end());
    
    vector<Particle> resample_particles;
    
    //For each particle
    int i;
    for (i = 0; i < num_particles; i++){
        
        resample_particles.push_back(particles[distribution(gen)]);
    }
    
    particles = resample_particles;
}


//----------------------------------------------------------------------------------


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

//----------------------------------------------------------------------------------


string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

//----------------------------------------------------------------------------------


string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

//----------------------------------------------------------------------------------


string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
