import add_src_path
import numpy as np
from boundaries import BoundaryWithCondition


class BoundaryLine(BoundaryWithCondition):
    def __init__(self, si, sf, vi, vf, *args, **kwargs):
        super(BoundaryLine, self).__init__(*args, **kwargs)
        self.si = si
        self.vi = vi
        self.ds = sf - si
        self.dv = vf - vi

    def evaluate(self, t):
        s = self.si + self.ds*t
        v = self.vi + self.dv*t
        return s, v


class BoundaryWaveX(BoundaryLine):
    def __init__(self, *args, amplitude=1, frequency=1, **kwargs):
        super(BoundaryWaveX, self).__init__(*args, **kwargs)
        self._amplitude = amplitude
        self._frequency = frequency * 2 * np.pi

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        import pudb
        pudb.start()
        sys.exit('')
        
    def evaluate(self, t):
        s, v = super(BoundaryWaveX, self).evaluate(t)
        dv = self.amplitude * np.sin(self.frequency * t)
        return np.array([s, v + dv])

    def __str__(self):
        return f'{self.si}, {self.vi}, {self.ds}, {self.dv}, {self.amplitude}, {self.frequency}'


class BoundaryWaveY(BoundaryLine):

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        import pudb
        pudb.start()
        sys.exit('')
        
    def __init__(self, *args, amplitude=1, frequency=1, **kwargs):
        super(BoundaryWaveY, self).__init__(*args, **kwargs)
        self._amplitude = amplitude
        self._frequency = frequency * 2 * np.pi
        
    def evaluate(self, t):
        s, v = super(BoundaryWaveY, self).evaluate(t)
        ds = self.amplitude * np.sin(self.frequency * t)
        return np.array([s + ds, v])
