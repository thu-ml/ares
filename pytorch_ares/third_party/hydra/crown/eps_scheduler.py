import numpy as np

 

class EpsilonScheduler():
    def __init__(self, schedule_type, init_step, final_step, init_value, final_value, num_steps_per_epoch, mid_point=.25, beta=4.):
        self.schedule_type = schedule_type
        self.init_step = init_step
        self.final_step = final_step
        self.init_value = init_value
        self.final_value = final_value
        self.mid_point = mid_point
        self.beta = beta
        self.num_steps_per_epoch = num_steps_per_epoch
        assert self.final_value >= self.init_value
        assert self.final_step >= self.init_step,\
            "{} should be larger than {}".format(self.final_step, self.init_step)
        assert self.beta >= 2.
        assert self.mid_point >= 0. and self.mid_point <= 1. 
    
    def get_eps(self, epoch, step):
        if self.schedule_type == "smoothed":
            return self.smooth_schedule(epoch * self.num_steps_per_epoch + step, self.init_step, self.final_step, self.init_value, self.final_value, self.mid_point, self.beta)
        else:
            return self.linear_schedule(epoch * self.num_steps_per_epoch + step, self.init_step, self.final_step, self.init_value, self.final_value)
    
    # Smooth schedule that slowly morphs into a linear schedule.
    # Code is adapted from DeepMind's IBP implementation:
    # https://github.com/deepmind/interval-bound-propagation/blob/2c1a56cb0497d6f34514044877a8507c22c1bd85/interval_bound_propagation/src/utils.py#L84
    def smooth_schedule(self, step, init_step, final_step, init_value, final_value, mid_point=.25, beta=4.):
        """Smooth schedule that slowly morphs into a linear schedule."""
        assert final_value >= init_value
        assert final_step >= init_step
        assert beta >= 2.
        assert mid_point >= 0. and mid_point <= 1.
        mid_step = int((final_step - init_step) * mid_point) + init_step 
        if mid_step <= init_step:
            alpha = 1.
        else:
            t = (mid_step - init_step) ** (beta - 1.)
            alpha = (final_value - init_value) / ((final_step - mid_step) * beta * t + (mid_step - init_step) * t)
        mid_value = alpha * (mid_step - init_step) ** beta + init_value
        is_ramp = float(step > init_step)
        is_linear = float(step >= mid_step)
        return (is_ramp * (
            (1. - is_linear) * (
                init_value +
                alpha * float(step - init_step) ** beta) +
            is_linear * self.linear_schedule(
                step, mid_step, final_step, mid_value, final_value)) +
                (1. - is_ramp) * init_value)
        
    # Linear schedule.
    # Code is adapted from DeepMind's IBP implementation:
    # https://github.com/deepmind/interval-bound-propagation/blob/2c1a56cb0497d6f34514044877a8507c22c1bd85/interval_bound_propagation/src/utils.py#L73 
    def linear_schedule(self, step, init_step, final_step, init_value, final_value):
        """Linear schedule."""
        assert final_step >= init_step
        if init_step == final_step:
            return final_value
        rate = float(step - init_step) / float(final_step - init_step)
        linear_value = rate * (final_value - init_value) + init_value
        return np.clip(linear_value, min(init_value, final_value), max(init_value, final_value))
