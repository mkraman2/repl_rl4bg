# pid_controller.py

class PIDController:
    def __init__(self, Kp, Ki, Kd, target_glucose=110):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target_glucose

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def compute_action(self, current_glucose):
        error = current_glucose - self.target
        self.integral += error
        derivative = 0 if self.prev_error is None else (error - self.prev_error)
        self.prev_error = error

        action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return [max(0.0, action)]  # ensure non-negative insulin dose
