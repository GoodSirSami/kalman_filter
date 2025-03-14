use nalgebra as na;
use std::f64;
use std::net::UdpSocket;
use std::thread;
use std::time::Duration;

// Define our state: position (x,y,z) and velocity (vx,vy,vz)
// For a complete physical model we could also add acceleration
type Vector6 = na::Vector6<f64>;
type Matrix6 = na::Matrix6<f64>;
type Vector3 = na::Vector3<f64>;
type Matrix3 = na::Matrix3<f64>;

struct KalmanFilter {
    // State vector: [x, y, z, vx, vy, vz]
    state: Vector6,
    // State transition matrix
    transition_matrix: Matrix6,
    // Process covariance matrix
    process_covariance: Matrix6,
    // State covariance matrix
    state_covariance: Matrix6,
    // Measurement matrix (maps state to measurement)
    measurement_matrix: na::Matrix3x6<f64>,
    // Measurement noise covariance
    measurement_noise: Matrix3,
    // Time step
    dt: f64,
}

impl KalmanFilter {
    fn new(dt: f64, process_noise: f64, measurement_noise: f64) -> Self {
        // Initial state: all zeros
        let state = Vector6::zeros();
        
        // State transition matrix (motion model)
        // For constant velocity model: x_t+1 = x_t + vx_t * dt, etc.
        let mut transition_matrix = Matrix6::identity();
        transition_matrix[(0, 3)] = dt; // x += vx*dt
        transition_matrix[(1, 4)] = dt; // y += vy*dt
        transition_matrix[(2, 5)] = dt; // z += vz*dt
        
        // Add gravity to z component (assuming negative z is downward)
        // For simplicity we're not modeling drag/air resistance
        
        // Process covariance (uncertainty in the model)
        let mut process_covariance = Matrix6::identity() * process_noise;
        // Higher uncertainty in velocity components and especially z due to gravity
        process_covariance[(5, 5)] *= 2.0; // Higher uncertainty in vz due to gravity
        
        // Measurement matrix - we only directly measure position (x,y,z)
        let mut measurement_matrix = na::Matrix3x6::<f64>::zeros();
        measurement_matrix[(0, 0)] = 1.0; // measure x
        measurement_matrix[(1, 1)] = 1.0; // measure y
        measurement_matrix[(2, 2)] = 1.0; // measure z
        
        // Measurement noise
        let measurement_noise = Matrix3::identity() * measurement_noise;
        
        // Initial state covariance (high uncertainty at first)
        let state_covariance = Matrix6::identity() * 1000.0;
        
        KalmanFilter {
            state,
            transition_matrix,
            process_covariance,
            state_covariance,
            measurement_matrix,
            measurement_noise,
            dt,
        }
    }
    
    fn predict(&mut self) {
        // Apply gravity to the state's z velocity component
        self.state[5] -= 9.81 * self.dt; // gravity affects vz
        
        // Predict next state
        self.state = self.transition_matrix * self.state;
        
        // Update state covariance
        self.state_covariance = self.transition_matrix * self.state_covariance * 
                               self.transition_matrix.transpose() + self.process_covariance;
    }
    
    fn update(&mut self, measurement: &Vector3) {
        // Calculate Kalman gain
        let measurement_prediction = self.measurement_matrix * self.state;
        
        let innovation = measurement - measurement_prediction;
        
        let innovation_covariance = self.measurement_matrix * self.state_covariance * 
                                   self.measurement_matrix.transpose() + self.measurement_noise;
        
        let kalman_gain = self.state_covariance * self.measurement_matrix.transpose() * 
                         innovation_covariance.try_inverse().unwrap();
        
        // Update state
        self.state += kalman_gain * innovation;
        
        // Update state covariance
        let identity = Matrix6::identity();
        self.state_covariance = (identity - kalman_gain * self.measurement_matrix) * self.state_covariance;
    }
    
    fn get_position(&self) -> Vector3 {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }
    
    fn get_velocity(&self) -> Vector3 {
        Vector3::new(self.state[3], self.state[4], self.state[5])
    }
}

// Represents a plane in 3D space using a point and normal vector
struct Plane {
    point: Vector3,
    normal: Vector3,
}

impl Plane {
    fn new(point: Vector3, normal: Vector3) -> Self {
        let normalized = normal.normalize();
        Plane {
            point,
            normal: normalized,
        }
    }
    
    // Find intersection of a ray with the plane
    fn intersect_with_ray(&self, ray_origin: &Vector3, ray_direction: &Vector3) -> Option<Vector3> {
        let denom = ray_direction.dot(&self.normal);
        
        // If ray is parallel to plane, no intersection
        if denom.abs() < 1e-6 {
            return None;
        }
        
        let v = &self.point - ray_origin;
        let t = v.dot(&self.normal) / denom;
        
        // If intersection is behind the ray, no intersection
        if t < 0.0 {
            return None;
        }
        
        // Calculate intersection point
        let intersection = ray_origin + ray_direction * t;
        Some(intersection)
    }
}

fn main() -> std::io::Result<()> {
    // Define floor plane (z=0, normal pointing up)
    let floor = Plane::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
    
    // Initialize Kalman filter
    let dt = 0.1; // Time step between measurements (adjust as needed)
    let process_noise = 0.01; // Process noise (model uncertainty)
    let measurement_noise = 0.1; // Measurement noise
    
    let mut kf = KalmanFilter::new(dt, process_noise, measurement_noise);
    let mut is_initialized = false;
    
    // Set up UDP socket for receiving measurements from Python
    let receive_socket = UdpSocket::bind("127.0.0.1:5005")?;
    println!("Listening for basketball position data on 127.0.0.1:5005");
    receive_socket.set_nonblocking(true)?;
    
    // Set up UDP socket for sending predictions
    let send_socket = UdpSocket::bind("127.0.0.1:5006")?;
    let target_address = "127.0.0.1:4321"; // Where to send predictions
    
    // Buffer to receive data
    let mut buf = [0u8; 12]; // 3 floats * 4 bytes each
    
    loop {
        // Try to receive data from Python
        match receive_socket.recv_from(&mut buf) {
            Ok((_, _)) => {
                // Parse the received data (3 floats: x, y, z)
                let x = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f64;
                let y = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as f64;
                let z = f32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as f64;
                
                let measurement = Vector3::new(x, y, z);
                println!("Received measurement: ({:.2}, {:.2}, {:.2})", x, y, z);
                
                if !is_initialized {
                    // Initialize state with first measurement
                    kf.state[0] = x;
                    kf.state[1] = y;
                    kf.state[2] = z;
                    is_initialized = true;
                    println!("Kalman filter initialized");
                } else {
                    // Predict next state
                    kf.predict();
                    
                    // Update with measurement
                    kf.update(&measurement);
                    
                    // Display current state
                    let position = kf.get_position();
                    let velocity = kf.get_velocity();
                    
                    println!("Filtered: Position = ({:.2}, {:.2}, {:.2})", 
                            position[0], position[1], position[2]);
                    println!("         Velocity = ({:.2}, {:.2}, {:.2})", 
                            velocity[0], velocity[1], velocity[2]);
                    
                    // Predict collision with floor
                    let mut collision_point = None;
                    let mut pos = position;
                    let mut vel = velocity;
                    
                    for _ in 0..50 {
                        // Apply gravity to velocity
                        vel[2] -= 9.81 * dt;
                        
                        // Calculate next position
                        let next_pos = pos + vel * dt;
                        
                        // Check if we crossed the floor plane
                        let direction = (next_pos - pos).normalize();
                        if let Some(intersection) = floor.intersect_with_ray(&pos, &direction) {
                            // Calculate distance to intersection
                            let distance = (intersection - pos).magnitude();
                            if distance <= vel.magnitude() * dt {
                                collision_point = Some(intersection);
                                break;
                            }
                        }
                        
                        // Stop if we've hit the floor
                        if next_pos[2] <= 0.0 {
                            collision_point = Some(Vector3::new(next_pos[0], next_pos[1], 0.0));
                            break;
                        }
                        
                        pos = next_pos;
                    }
                    
                    // Send collision point via UDP if available
                    if let Some(collision) = collision_point {
                        println!("Predicted collision: ({:.2}, {:.2}, {:.2})", 
                                collision[0], collision[1], collision[2]);
                                
                        // Convert to bytes (3 f32 values)
                        let mut data = [0u8; 12];
                        data[0..4].copy_from_slice(&(collision[0] as f32).to_le_bytes());
                        data[4..8].copy_from_slice(&(collision[1] as f32).to_le_bytes());
                        data[8..12].copy_from_slice(&(collision[2] as f32).to_le_bytes());
                        
                        // Send the prediction
                        send_socket.send_to(&data, target_address)?;
                    }
                }
            },
            Err(e) => {
                // No data available or an error occurred
                if e.kind() != std::io::ErrorKind::WouldBlock {
                    eprintln!("Error receiving data: {}", e);
                }
                // Small sleep to prevent CPU from maxing out
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
