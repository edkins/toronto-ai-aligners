import itertools
import random
import numpy as np
from PIL import Image

def values_to_rgb(values):
    intensity = np.minimum(np.log1p(np.abs(values)), 1)
    pos = np.maximum(0, np.sign(values))
    neg = np.maximum(0, -np.sign(values))
    r = ((1 - intensity * neg) * 255).astype('uint8')
    g = ((1 - intensity) * 255).astype('uint8')
    b = ((1 - intensity * pos) * 255).astype('uint8')
    result = np.stack([r,g,b], axis=2)
    return result

def shape_indices(shape):
    return list(itertools.product(*[range(d) for d in shape]))

def choose_segments(model, max_per_tensor):
    state = model.state_dict()
    result = []
    for key, value in state.items():
        indices = shape_indices(value.shape)
        if len(indices) >= max_per_tensor:
            indices = random.sample(indices, k=max_per_tensor)
        indices = np.array(indices, dtype='int32')
        result.append((key, indices))
    return result

def segments_width(segments):
    result = 0
    for _, indices in segments:
        result += len(indices)
    return result

class Imager:
    def __init__(self, filename, *, width=None, segments=None):
        if width == None:
            width = segments_width(segments)
        self.im = Image.new('RGB', (width,0), (255,255,255))
        self.filename = filename
        self.segments = segments

    def extend(self, values):
        rgb = values_to_rgb(values)
        w0, h0 = self.im.size
        w1, h1, _ = rgb.shape
        if w0 != w1:
            raise Exception(f"Width {w1} does not match image width {w0}")
        extra = Image.fromarray(rgb, 'RGB').transpose(Image.Transpose.TRANSPOSE)
        new_im = Image.new('RGB', (w0, h0 + h1))
        new_im.paste(self.im, (0,0))
        new_im.paste(extra, (0,h0))
        self.im = new_im

    def save(self):
        self.im.save(self.filename)

    def extend_from_model(self, model):
        state = model.state_dict()
        values = np.zeros((self.im.width,1), dtype='float32')
        offset = 0
        for key, indices in self.segments:
            #print(state[key].shape)
            #print(indices.min(axis=0), indices.max(axis=0))
            if indices.shape[1] == 1:
                values[offset:offset+len(indices),0] = state[key][indices[:,0]].cpu().numpy()
            elif indices.shape[1] == 2:
                values[offset:offset+len(indices),0] = state[key][indices[:,0],indices[:,1]].cpu().numpy()
            elif indices.shape[1] == 3:
                values[offset:offset+len(indices),0] = state[key][indices[:,0],indices[:,1],indices[:,2]].cpu().numpy()
            else:
                raise Exception()
            offset += len(indices)
        self.extend(values)