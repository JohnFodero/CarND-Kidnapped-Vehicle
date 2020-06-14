/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  for(int i = 0; i < num_particles; ++i) {
    Particle new_particle;
    new_particle.id = i;
    new_particle.x = dist_x(gen);
    new_particle.y = dist_y(gen);
    new_particle.theta = dist_theta(gen);
    new_particle.weight = 1;
    particles.push_back(new_particle);
    weights.push_back(0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   std::default_random_engine gen;
   std::normal_distribution<double> dist_x(0, std_pos[0]);
   std::normal_distribution<double> dist_y(0, std_pos[1]);
   std::normal_distribution<double> dist_theta(0, std_pos[2]);
   for(int i = 0; i < num_particles; ++i) {
     if(fabs(yaw_rate) < 0.00001) {
       particles[i].x += velocity*delta_t*std::cos(particles[i].theta);
       particles[i].y += velocity*delta_t*std::sin(particles[i].theta);
     }
     else {
       particles[i].x += (velocity/yaw_rate)*(std::sin(particles[i].theta + yaw_rate*delta_t) - std::sin(particles[i].theta));
       particles[i].y += (velocity/yaw_rate)*(std::cos(particles[i].theta) - std::cos(particles[i].theta + yaw_rate*delta_t));
       particles[i].theta += yaw_rate*delta_t;
     }
     particles[i].x += dist_x(gen);
     particles[i].y += dist_y(gen);
     particles[i].theta += dist_theta(gen);
   }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  //UNUSED
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
   double total_weight = 0;
   double weight;
   for(int i = 0; i < num_particles; ++i) {
     vector<LandmarkObs> predicted;
     for(int h = 0; h < observations.size(); ++h) {
       LandmarkObs temp;
       double x_particle = particles[i].x;
       double y_particle = particles[i].y;
       double x_car = observations[h].x;
       double y_car = observations[h].y;
       double theta = particles[i].theta;
       temp.x = x_car*std::cos(theta) - y_car*std::sin(theta) + x_particle;
       temp.y = x_car*std::sin(theta) + y_car*std::cos(theta) + y_particle;

       temp.id = -1;
       double min_dist = 100000;
       double temp_dist;
       for(int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
         temp_dist = dist(temp.x, temp.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
         if(temp_dist < min_dist) {
           temp.id = j;
           min_dist = temp_dist;
         }
       }
       predicted.push_back(temp);
     }

     weight = 0;
     for(int j = 0; j < predicted.size(); ++j) {

       int id = predicted[j].id;
       if(id == -1) {
         continue;
       }
       double std_x = std_landmark[0];
       double std_y = std_landmark[1];
       double mu_x = map_landmarks.landmark_list[id].x_f;
       double mu_y = map_landmarks.landmark_list[id].y_f;
       double x = predicted[j].x;
       double y = predicted[j].y;
       double diff_x = x - mu_x;
       double diff_y = y - mu_y;
       weight += (1/(2*M_PI*std_x*std_y))*std::exp(-1*(((diff_x * diff_x)/(2*std_x*std_x)) + ((diff_y * diff_y)/(2*std_y*std_y))));
     }
     particles[i].weight = weight;
     weights[i] = weight;
     total_weight += weight;

   }

   for(int i = 0; i < num_particles; ++i) {
     particles[i].weight /= total_weight;
     weights[i] /= total_weight;
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> new_particles;
  std::default_random_engine gen;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  for(int i = 0; i < num_particles; ++i) {
    new_particles.push_back(particles[d(gen)]);
  }
  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
