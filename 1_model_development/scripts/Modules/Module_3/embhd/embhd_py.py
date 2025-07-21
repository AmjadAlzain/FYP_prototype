"""
EmbHD Python Wrapper - Hyperdimensional Computing for ESP32-S3-EYE

This module provides Python bindings for the EmbHD C library, enabling
training and testing in Python before deployment to the ESP32.
"""

import os
import numpy as np
import ctypes
from enum import IntEnum
from pathlib import Path
import torch

# Vector types matching C implementation
class EmbHDVectorType(IntEnum):
    BINARY = 0
    BIPOLAR = 1
    FLOAT = 2

class EmbHDVector:
    """Python implementation of embhd_vector_t from C library."""
    
    def __init__(self, dim, vtype=EmbHDVectorType.BIPOLAR):
        """Initialize a vector with specified dimensions and type."""
        self.dim = dim
        self.vtype = vtype
        
        if vtype == EmbHDVectorType.BINARY:
            # For binary vectors, pack bits (1 byte per 8 values)
            self.data = np.zeros(((dim + 7) // 8,), dtype=np.uint8)
        elif vtype == EmbHDVectorType.BIPOLAR:
            # For bipolar vectors, use int8 (-1/+1)
            self.data = np.zeros((dim,), dtype=np.int8)
        elif vtype == EmbHDVectorType.FLOAT:
            # For float vectors, use float32
            self.data = np.zeros((dim,), dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, array, vtype=None):
        """Create a vector from a numpy array."""
        if vtype is None:
            # Infer type from array
            if array.dtype == np.bool_ or array.dtype == np.uint8:
                vtype = EmbHDVectorType.BINARY
            elif array.dtype == np.int8:
                vtype = EmbHDVectorType.BIPOLAR
            else:
                vtype = EmbHDVectorType.FLOAT
        
        vec = cls(len(array), vtype)
        
        if vtype == EmbHDVectorType.BINARY:
            # Pack bits
            for i in range(len(array)):
                if array[i] > 0:
                    byte_idx = i // 8
                    bit_idx = i % 8
                    vec.data[byte_idx] |= (1 << bit_idx)
        else:
            vec.data = array.astype(vec.data.dtype)
        
        return vec
    
    def to_numpy(self):
        """Convert to numpy array."""
        if self.vtype == EmbHDVectorType.BINARY:
            # Unpack bits
            result = np.zeros((self.dim,), dtype=np.uint8)
            for i in range(self.dim):
                byte_idx = i // 8
                bit_idx = i % 8
                result[i] = (self.data[byte_idx] >> bit_idx) & 1
            return result
        else:
            return self.data
    
    def to_torch(self):
        """Convert to PyTorch tensor."""
        return torch.from_numpy(self.to_numpy())


class EmbHDProjection:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.normal(0, 1.0 / np.sqrt(in_features),
                                        (in_features, out_features)).astype(np.float32)
        # Add placeholders for quantized data
        self.weights_quantized = None
        self.scale = None
        self.zero_point = None

    def quantize(self):
        """Quantizes the FP32 weights to INT8 and calculates scale/zero-point."""
        if self.weights is None:
            raise ValueError("Cannot quantize empty weights.")

        print("Quantizing projection matrix from FP32 to INT8...")
        min_val, max_val = self.weights.min(), self.weights.max()

        self.scale = (max_val - min_val) / 255.0
        # Ensure scale is not zero
        if self.scale == 0.0:
            self.scale = 1.0

        self.zero_point = -min_val / self.scale
        self.zero_point = int(np.round(self.zero_point))

        quantized = np.round(self.weights / self.scale + self.zero_point)
        self.weights_quantized = np.clip(quantized, -128, 127).astype(np.int8)
        print("Quantization complete.")
    
    @classmethod
    def from_numpy(cls, weights):
        """Create a projection from a numpy array."""
        proj = cls(weights.shape[0], weights.shape[1])
        proj.weights = weights.astype(np.float32)
        return proj
    
    @classmethod
    def from_torch(cls, weights):
        """Create a projection from a PyTorch tensor."""
        return cls.from_numpy(weights.detach().cpu().numpy())
    
    def to_numpy(self):
        """Convert to numpy array."""
        return self.weights
    
    def to_torch(self):
        """Convert to PyTorch tensor."""
        return torch.from_numpy(self.weights)

    def to_c_header(self, variable_name="proj_matrix"):
        header = f"// EmbHD Projection Matrix\n"
        header += f"#define EMBHD_IN_FEATURES {self.in_features}\n"
        header += f"#define EMBHD_OUT_FEATURES {self.out_features}\n\n"

        # --- UPDATED: Export INT8 version if available ---
        if self.weights_quantized is not None:
            header += "// Quantized INT8 Projection Matrix\n"
            header += f'const float {variable_name}_scale = {self.scale:.8f}f;\n'
            header += f'const int8_t {variable_name}_zero_point = {self.zero_point};\n\n'
            header += f"const int8_t {variable_name}[EMBHD_IN_FEATURES][EMBHD_OUT_FEATURES] = {{\n"
            for i in range(self.in_features):
                header += "    {" + ", ".join(map(str, self.weights_quantized[i])) + "}"
                if i < self.in_features - 1: header += ","
                header += "\n"
        else:  # Fallback to float if not quantized
            header += "// Standard FP32 Projection Matrix\n"
            header += f"const float {variable_name}[EMBHD_IN_FEATURES][EMBHD_OUT_FEATURES] = {{\n"
            for i in range(self.in_features):
                header += "    {" + ", ".join(f"{x:.6f}f" for x in self.weights[i]) + "}"
                if i < self.in_features - 1: header += ","
                header += "\n"

        header += "};\n"
        return header


class EmbHDPrototypes:
    """Python implementation of embhd_prototypes_t from C library."""
    
    def __init__(self, num_classes, dim, vtype=EmbHDVectorType.BIPOLAR):
        """Initialize prototypes for a specified number of classes."""
        self.num_classes = num_classes
        self.dim = dim
        self.vtype = vtype
        self.vectors = [EmbHDVector(dim, vtype) for _ in range(num_classes)]
    
    @classmethod
    def from_numpy(cls, arrays, vtype=EmbHDVectorType.BIPOLAR):
        """Create prototypes from a list of numpy arrays."""
        if len(arrays) == 0:
            return None
        
        protos = cls(len(arrays), len(arrays[0]), vtype)
        for i, array in enumerate(arrays):
            protos.vectors[i] = EmbHDVector.from_numpy(array, vtype)
        
        return protos
    
    @classmethod
    def from_torch(cls, tensors, vtype=EmbHDVectorType.BIPOLAR):
        """Create prototypes from a list of PyTorch tensors."""
        return cls.from_numpy([t.detach().cpu().numpy() for t in tensors], vtype)
    
    def to_numpy(self):
        """Convert to list of numpy arrays."""
        return [vec.to_numpy() for vec in self.vectors]
    
    def to_torch(self):
        """Convert to list of PyTorch tensors."""
        return [torch.from_numpy(vec.to_numpy()) for vec in self.vectors]
    
    def to_c_header(self, variable_name="prototypes"):
        """Generate C header file content for these prototypes."""
        header = f"// EmbHD Class Prototypes ({self.num_classes} classes, {self.dim} dimensions)\n"
        header += f"#define EMBHD_NUM_CLASSES {self.num_classes}\n"
        header += f"#define EMBHD_VECTOR_DIM {self.dim}\n\n"
        
        if self.vtype == EmbHDVectorType.BINARY:
            dtype = "uint8_t"
            header += f"// Binary vectors (bits packed, 8 per byte)\n"
            byte_count = (self.dim + 7) // 8
            header += f"const {dtype} {variable_name}[EMBHD_NUM_CLASSES][{byte_count}] = {{\n"
            
            for i in range(self.num_classes):
                header += "    {"
                for j in range(byte_count):
                    header += f"0x{self.vectors[i].data[j]:02x}"
                    if j < byte_count - 1:
                        header += ", "
                header += "}"
                if i < self.num_classes - 1:
                    header += ","
                header += "\n"
        
        elif self.vtype == EmbHDVectorType.BIPOLAR:
            dtype = "int8_t"
            header += f"// Bipolar vectors (-1/+1)\n"
            header += f"const {dtype} {variable_name}[EMBHD_NUM_CLASSES][EMBHD_VECTOR_DIM] = {{\n"
            
            for i in range(self.num_classes):
                header += "    {"
                for j in range(self.dim):
                    header += f"{self.vectors[i].data[j]}"
                    if j < self.dim - 1:
                        header += ", "
                header += "}"
                if i < self.num_classes - 1:
                    header += ","
                header += "\n"
        
        else:  # FLOAT
            dtype = "float"
            header += f"// Float vectors\n"
            header += f"const {dtype} {variable_name}[EMBHD_NUM_CLASSES][EMBHD_VECTOR_DIM] = {{\n"
            
            for i in range(self.num_classes):
                header += "    {"
                for j in range(self.dim):
                    header += f"{self.vectors[i].data[j]:.6f}"
                    if j < self.dim - 1:
                        header += ", "
                header += "}"
                if i < self.num_classes - 1:
                    header += ","
                header += "\n"
        
        header += "};\n"
        return header


class EmbHDModel:
    """Python implementation of embhd_model_t from C library."""
    
    def __init__(self, in_features, num_classes, hd_dim=20000, vtype=EmbHDVectorType.BIPOLAR):
        """Initialize a full HDC model."""
        self.projection = EmbHDProjection(in_features, hd_dim)
        self.prototypes = EmbHDPrototypes(num_classes, hd_dim, vtype)
        self.sample_counts = np.zeros(num_classes, dtype=np.int32)
    
    def encode(self, features):
        """Encode a feature vector (or batch) using projection matrix."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # Handle single feature vector or batch
        is_batch = len(features.shape) > 1
        if not is_batch:
            features = features.reshape(1, -1)
        
        # Project features
        projected = np.matmul(features, self.projection.weights.T)
        
        # Create vectors based on type
        result = []
        for i in range(projected.shape[0]):
            if self.prototypes.vtype == EmbHDVectorType.BINARY:
                # Binarize (0/1)
                data = (projected[i] > 0).astype(np.uint8)
            elif self.prototypes.vtype == EmbHDVectorType.BIPOLAR:
                # Bipolarize (-1/+1)
                data = np.where(projected[i] > 0, 1, -1).astype(np.int8)
            else:  # FLOAT
                data = projected[i]
            
            result.append(EmbHDVector.from_numpy(data, self.prototypes.vtype))
        
        return result[0] if not is_batch else result
    
    def classify(self, query, return_scores=False):
        """Classify a query vector against class prototypes."""
        if isinstance(query, np.ndarray) or isinstance(query, torch.Tensor):
            # Encode if raw features provided
            query = self.encode(query)
        
        scores = np.zeros(self.prototypes.num_classes)
        query_data = query.to_numpy()
        
        for i in range(self.prototypes.num_classes):
            proto_data = self.prototypes.vectors[i].to_numpy()
            
            if self.prototypes.vtype == EmbHDVectorType.BINARY:
                # Hamming similarity (proportion of matching bits)
                scores[i] = np.mean(query_data == proto_data)
            
            elif self.prototypes.vtype == EmbHDVectorType.BIPOLAR:
                # Dot product for bipolar vectors
                scores[i] = np.sum(query_data * proto_data) / len(query_data)
            
            else:  # FLOAT
                # Cosine similarity
                norm_q = np.linalg.norm(query_data)
                norm_p = np.linalg.norm(proto_data)
                if norm_q > 0 and norm_p > 0:
                    scores[i] = np.dot(query_data, proto_data) / (norm_q * norm_p)
        
        predicted_class = np.argmax(scores)
        
        return (predicted_class, scores) if return_scores else predicted_class
    
    def predict(self, features, return_scores=False):
        """Predict class for input features."""
        encoded = self.encode(features)
        # If encode returns a list (from a batch of size 1), extract the single item
        if isinstance(encoded, list):
            encoded = encoded[0]
        return self.classify(encoded, return_scores)
    
    def train_sample(self, features, class_id, learning_rate=0.1):
        """Train model on a single example."""
        if class_id >= self.prototypes.num_classes:
            raise ValueError(f"Class ID {class_id} out of range")
        
        # Encode the features
        encoded = self.encode(features)
        
        # Get the prototype to update
        proto = self.prototypes.vectors[class_id]
        
        # Update prototype based on vector type
        if self.prototypes.vtype == EmbHDVectorType.BINARY:
            # For binary vectors, use probabilistic update
            encoded_data = encoded.to_numpy()
            proto_data = proto.to_numpy()
            
            # Determine positions to update (with probability = learning_rate)
            update_mask = np.random.rand(len(proto_data)) < learning_rate
            update_mask = update_mask & (encoded_data != proto_data)
            
            # Apply updates
            if np.any(update_mask):
                new_data = proto_data.copy()
                new_data[update_mask] = encoded_data[update_mask]
                self.prototypes.vectors[class_id] = EmbHDVector.from_numpy(new_data, EmbHDVectorType.BINARY)
        
        elif self.prototypes.vtype == EmbHDVectorType.BIPOLAR:
            # For bipolar vectors, weighted average then quantize
            encoded_data = encoded.to_numpy()
            proto_data = proto.to_numpy()
            
            # Weighted average
            updated = (1.0 - learning_rate) * proto_data + learning_rate * encoded_data
            
            # Quantize back to -1/+1
            updated = np.where(updated > 0, 1, -1).astype(np.int8)
            
            self.prototypes.vectors[class_id] = EmbHDVector.from_numpy(updated, EmbHDVectorType.BIPOLAR)
        
        else:  # FLOAT
            # For float vectors, simple weighted average
            encoded_data = encoded.to_numpy()
            proto_data = proto.to_numpy()
            
            updated = (1.0 - learning_rate) * proto_data + learning_rate * encoded_data
            
            self.prototypes.vectors[class_id] = EmbHDVector.from_numpy(updated, EmbHDVectorType.FLOAT)
        
        # Update sample count
        self.sample_counts[class_id] += 1
    
    def train_batch(self, features, labels, learning_rate=0.1):
        """Train model on a batch of examples."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Handle single example or batch
        is_batch = len(features.shape) > 1
        if not is_batch:
            features = features.reshape(1, -1)
            labels = np.array([labels])
        
        # Train on each example
        for i in range(len(labels)):
            self.train_sample(features[i], labels[i], learning_rate)
    
    def export_c_headers(self, output_dir, proj_name="proj_matrix", proto_name="prototypes"):
        """Export model to C header files for ESP32."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Export projection matrix
        proj_header = self.projection.to_c_header(proj_name)
        with open(output_dir / "embhd_projection.h", "w") as f:
            f.write(proj_header)
        
        # Export prototypes
        proto_header = self.prototypes.to_c_header(proto_name)
        with open(output_dir / "embhd_prototypes.h", "w") as f:
            f.write(proto_header)
        
        print(f"Exported EmbHD model to C headers in {output_dir}")
        return output_dir / "embhd_projection.h", output_dir / "embhd_prototypes.h"
    
    @classmethod
    def from_torch_hd(cls, torch_hd_model, projection_matrix, hd_dim=20000):
        """Convert a torch_hd model to EmbHD."""
        if hasattr(torch_hd_model, 'prototypes'):
            num_classes = torch_hd_model.prototypes.shape[0]
            in_features = projection_matrix.shape[0]
            
            # Create EmbHD model
            model = cls(in_features, num_classes, hd_dim, EmbHDVectorType.BIPOLAR)
            
            # Set projection matrix
            model.projection = EmbHDProjection.from_numpy(projection_matrix.detach().cpu().numpy())
            
            # Set prototypes
            protos_np = torch_hd_model.prototypes.detach().cpu().numpy()
            # Convert to bipolar (-1/+1)
            protos_bipolar = np.where(protos_np > 0, 1, -1).astype(np.int8)
            model.prototypes = EmbHDPrototypes.from_numpy(protos_bipolar, EmbHDVectorType.BIPOLAR)
            
            return model
        else:
            raise ValueError("Unsupported torch_hd model format")
    
    def save(self, path):
        """Save model to file."""
        data = {
            'projection': self.projection.to_numpy(),
            'prototypes': [v.to_numpy() for v in self.prototypes.vectors],
            'vtype': int(self.prototypes.vtype),
            'sample_counts': self.sample_counts
        }
        np.savez(path, **data)
    
    @classmethod
    def load(cls, path):
        """Load model from file."""
        data = np.load(path, allow_pickle=True)
        
        # Correctly get dimensions from the saved projection matrix
        in_features = data['projection'].shape[0]
        hd_dim = data['projection'].shape[1]
        
        num_classes = len(data['prototypes'])
        vtype = EmbHDVectorType(int(data['vtype']))
        
        model = cls(in_features, num_classes, hd_dim, vtype)
        model.projection = EmbHDProjection.from_numpy(data['projection'])
        model.prototypes = EmbHDPrototypes.from_numpy(data['prototypes'], vtype)
        model.sample_counts = data['sample_counts']
        
        # Ensure the projection matrix has the correct dimensions
        model.projection.in_features = data['projection'].shape[0]
        model.projection.out_features = data['projection'].shape[1]
        
        return model
