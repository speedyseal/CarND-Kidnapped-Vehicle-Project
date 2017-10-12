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

static double particleProbability(const Particle &particle, double sig_x, double sig_y, const Map &map) {
	double cumprod = 1.;
	const double normterm = 1. / (2. * M_PI * sig_x * sig_y);
	for (int i = 0; i < particle.associations.size(); ++i) {
		const auto landmark_ind = particle.associations[i]-1;
		const auto &landmark = map.landmark_list[landmark_ind];
		// cout << "landmark_ind:" << landmark_ind << ", landmark_id: " << landmark.id_i << endl;
		const auto xm = landmark.x_f;
		const auto ym = landmark.y_f;
		const auto xs = particle.sense_x[i];
		const auto ys = particle.sense_y[i];
		cumprod *= normterm * exp(-((xs-xm)*(xs-xm)/(2.*sig_x*sig_x) + (ys-ym)*(ys-ym)/(2.*sig_y*sig_y)));
	}
	return cumprod;
}

static void translateObservations(Particle &particle, const std::vector<LandmarkObs>& observations) {
	particle.associations.resize(observations.size(), 0.);
	particle.sense_x.resize(observations.size(), 0.);
	particle.sense_y.resize(observations.size(), 0.);

	const auto xp = particle.x;
	const auto yp = particle.y;
	const auto theta = particle.theta;
	// cout << "\t---------" << "particle x,y, theta: " << xp << ", " << yp << ", " << theta << endl;
	for (int i = 0; i < observations.size(); ++i) {
		const auto xo = observations[i].x;
		const auto yo = observations[i].y;
		auto xt = xp + xo*cos(theta) - yo*sin(theta);
		auto yt = yp + xo*sin(theta) + yo*cos(theta);
		particle.sense_x[i] = xt;
		particle.sense_y[i] = yt;

		// cout << "xobs: " << xo << ", yobs: " << yo << ", xt: " << xt << ", yt: " << yt << endl;
	}
}

static double euclidean_distance(double x1, double y1, double x2, double y2) {
	const auto term1 = x1 - x2;
	const auto term2 = y1 - y2;
	// sqrt( (x1-x2)^2 + (y1-y2)^2)
	return sqrt(term1*term1 + term2*term2);
}

static void associateObservations(Particle &particle, const Map &map_landmarks) {
	const auto &landmark_list = map_landmarks.landmark_list;
	for (int i = 0; i < particle.associations.size(); ++i) {
		const auto xobs = particle.sense_x[i];
		const auto yobs = particle.sense_y[i];
		auto minInd = size_t(0);
		auto minDist = euclidean_distance(xobs, yobs, landmark_list[0].x_f, landmark_list[0].y_f);
		// cout << "xobs: " << xobs << ", yobs: " << yobs << ", xl0:" << landmark_list[0].x_f << ", yl0:" << landmark_list[0].y_f 
		// 	<< ", minDist: " << minDist << ", obsDist: ";
		for (int j = 1; j < landmark_list.size(); ++j) {
			const auto &landmark = landmark_list[j];
			const auto obsDist = euclidean_distance(xobs, yobs, landmark.x_f, landmark.y_f);
			// cout << obsDist << ", ";
			if (obsDist < minDist) {
				minDist = obsDist;
				minInd = j;
			}
		}
		// cout << endl << "minDist:" << minDist << endl;
		particle.associations[i] = landmark_list[minInd].id_i;
	}
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	const auto std_x = std[0];
	const auto std_y = std[1];
	const auto std_theta = std[2];

	// create normal (Gaussian) distributions given input parameters
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	particles.resize(num_particles);

	for (auto &particle : particles) {
		 const auto sample_x = dist_x(gen);
		 const auto sample_y = dist_y(gen);
		 const auto sample_theta = dist_theta(gen);

		 particle.x = sample_x;
		 particle.y = sample_y;
		 particle.theta = sample_theta;
		 particle.weight = 1.0;

	}

	weights.resize(num_particles, 1.0);	// set all weights to 1.0

	// for (const auto &particle : particles) {
	// 	cout << particle.x << ", " << particle.y << ", " << particle.theta << endl;
	// }
	// cout << "init" << endl;
	// cout << "weight size:" << weights.size() << endl;
	// std::copy(weights.begin(), weights.end(), std::ostream_iterator<float>(std::cout, " "));
	// cout << endl;
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// create normal (Gaussian) distributions given input parameters
	const auto std_x = std_pos[0];
	const auto std_y = std_pos[1];
	const auto std_theta = std_pos[2];

	default_random_engine gen;
	normal_distribution<double> dist_x(0., std_x);
	normal_distribution<double> dist_y(0., std_y);
	normal_distribution<double> dist_theta(0., std_theta);

	// cout << "prediction" << endl;
	for (auto &particle : particles) {
		 const auto sample_x = dist_x(gen);
		 const auto sample_y = dist_y(gen);
		 const auto sample_theta = dist_theta(gen);

		 const auto x0 = particle.x;
		 const auto y0 = particle.y;
		 const auto theta0 = particle.theta;

		 if (yaw_rate == 0.) {
			 particle.x = x0 + velocity * delta_t * cos(theta0) + sample_x;
			 particle.y = y0 + velocity * delta_t * sin(theta0) + sample_y;
			 particle.theta = theta0 + sample_theta;
		 } else {
			 particle.x = x0 + velocity / yaw_rate * (sin(theta0 + yaw_rate*delta_t) - sin(theta0) ) + sample_x;
			 particle.y = y0 + velocity / yaw_rate * (cos(theta0) - cos(theta0 + yaw_rate*delta_t) ) + sample_y;
			 particle.theta = theta0 + yaw_rate * delta_t + sample_theta;
		}
		while (particle.theta < 0.) {
			particle.theta += 2.*M_PI;
		}
		while (particle.theta > 2.*M_PI) {
			particle.theta -= 2.*M_PI;
		}
		// cout << particle.x << ", " << particle.y << ", " << particle.theta << endl;
	}
	// cout << "prediction:" << endl;
	// auto count = 0;
	// for (const auto &particle : particles) {
	//   cout << count++ << ": " <<particle.x << ", " << particle.y << ", " << particle.theta << endl;
	// }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

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
	weights.clear();
	const auto sig_x = std_landmark[0];
	const auto sig_y = std_landmark[1];
	for (auto &particle : particles) {
		translateObservations(particle, observations);	// stores transformed coordinates in xsense, ysense vectors per particle
		associateObservations(particle, map_landmarks);

		// cout << "associations: " << endl;
		// std::copy(particle.associations.begin(), particle.associations.end(), std::ostream_iterator<int>(std::cout, " "));
		// cout << endl;

		// cout << "xsense: " << endl;
		// std::copy(particle.sense_x.begin(), particle.sense_x.end(), std::ostream_iterator<float>(std::cout, " "));
		// cout << endl;

		// cout << "ysense: " << endl;
		// std::copy(particle.sense_y.begin(), particle.sense_y.end(), std::ostream_iterator<float>(std::cout, " "));
		// cout << endl;
		const auto particleProb = particleProbability(particle, sig_x, sig_y, map_landmarks);
		particle.weight = particleProb;
		weights.push_back(particleProb);
	}
	// cout << "updateWeights: " << weights.size() << endl;
	// std::copy(weights.begin(), weights.end(), std::ostream_iterator<float>(std::cout, " "));
	// cout << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> new_particles;

	for (auto &particle : particles) {
		new_particles.push_back(particles[d(gen)]);
	}
	particles = new_particles;

	// cout << "resample:" << endl;
	// auto count = 0;
	// for (const auto &particle : particles) {
	//   cout << count++ << ": " <<particle.x << ", " << particle.y << ", " << particle.theta << endl;
	// }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
