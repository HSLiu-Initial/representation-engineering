from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from itertools import islice

### Util Functions ###
def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    # TODO: should we require direction vectors to be unit vectors? then return H.dot(direction)
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return H.dot(direction) / mag

def recenter(x, mean=None):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    return x - mean

class RepReader(ABC):
    """Class to identify and store concept directions.
    
    Subclasses implement the abstract methods to identify concept directions 
    for each hidden layer via strategies including PCA, embedding vectors 
    (aka the logits method), and cluster means.

    RepReader instances are used by RepReaderPipeline to get concept scores.

    Directions can be used for downstream interventions."""

    @abstractmethod
    def __init__(self) -> None:
        self.direction_method = None
        self.directions = None # directions accessible via directions[layer][component_index]
        self.direction_signs = None # direction of high concept scores (mapping min/max to high/low)

    @abstractmethod
    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        """Get concept directions for each hidden layer of the model
        
        Args:
            model: Model to get directions for
            tokenizer: Tokenizer to use
            hidden_states: Hidden states of the model on the training data (per layer)
            hidden_layers: Layers to consider

        Returns:
            directions: A dict mapping layers to direction arrays (n_components, hidden_size)
        """
        pass 

    def get_signs(self, hidden_states, train_choices, hidden_layers):
        """Given labels for the training data hidden_states, determine whether the
        negative or positive direction corresponds to low/high concept 
        (and return corresponding signs -1 or 1 for each layer and component index)
        
        NOTE: This method assumes that there are 2 entries in hidden_states per label, 
        aka len(hidden_states[layer]) == 2 * len(train_choices). For example, if 
        n_difference=1, then hidden_states here should be the raw hidden states
        rather than the relative (i.e. the differences between pairs of examples).

        Args:
            hidden_states: Hidden states of the model on the training data (per layer)
            train_choices: Labels for the training data
            hidden_layers: Layers to consider

        Returns:
            signs: A dict mapping layers to sign arrays (n_components,)
        """        
        signs = {}

        if self.needs_hiddens and hidden_states is not None and len(hidden_states) > 0:
            for layer in hidden_layers:    
                assert hidden_states[layer].shape[0] == 2 * len(train_choices), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(train_choices)})"
                
                signs[layer] = []
                for component_index in range(self.n_components):
                    transformed_hidden_states = project_onto_direction(hidden_states[layer], self.directions[layer][component_index])
                    projected_scores = [transformed_hidden_states[i:i+2] for i in range(0, len(transformed_hidden_states), 2)]

                    outputs_min = [1 if min(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    outputs_max = [1 if max(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    
                    signs[layer].append(-1 if np.mean(outputs_min) > np.mean(outputs_max) else 1)
        else:
            for layer in hidden_layers:    
                signs[layer] = [1 for _ in range(self.n_components)]

        return signs


    def transform(self, hidden_states, hidden_layers, component_index):
        """Project the hidden states onto the concept directions in self.directions

        Args:
            hidden_states: dictionary with entries of dimension (n_examples, hidden_size)
            hidden_layers: list of layers to consider
            component_index: index of the component to use from self.directions

        Returns:
            transformed_hidden_states: dictionary with entries of dimension (n_examples,)
        """

        assert component_index < self.n_components

        transformed_hidden_states = {}
        for layer in hidden_layers:
            layer_hidden_states = hidden_states[layer]

            if hasattr(self, 'H_train_means'):
                layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer])

            # project hidden states onto found concept directions (e.g. onto PCA comp 0) 
            H_transformed = project_onto_direction(layer_hidden_states, self.directions[layer][component_index])
            transformed_hidden_states[layer] = H_transformed

        return transformed_hidden_states

class PCARepReader(RepReader):
    """Extract directions via PCA"""
    needs_hiddens = True 

    def __init__(self, n_components=1):
        super().__init__()
        self.n_components = n_components # 就是n_difference
        self.H_train_means = {}

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        """Get PCA components for each layer"""
        directions = {}

        for layer in hidden_layers:
            H_train = np.array(hidden_states[layer]) # 提取每一层的差距向量们 shape(train/2, 4096)

            H_train_mean = H_train.mean(axis=0, keepdims=True) # shape(1, 4096) 对每一列求均值，其实就是对差异向量跨样本求均值
            self.H_train_means[layer] = H_train_mean
            H_train = recenter(H_train, mean=H_train_mean) # 减去均值【PCA的要求】

            pca_model = PCA(n_components=self.n_components, whiten=False).fit(H_train) # 对于每一层都求了一个PCA whiten=False表示不对数据进行标准化处理【方差不是1】

            directions[layer] = pca_model.components_ # shape (n_components, n_features)
            self.n_components = pca_model.n_components_
        
        return directions

    def get_signs(self, hidden_states, train_labels, hidden_layers):

        signs = {}

        for layer in hidden_layers:
            assert hidden_states[layer].shape[0] == len(np.concatenate(train_labels)), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})"
            layer_hidden_states = hidden_states[layer]

            # NOTE: since scoring is ultimately comparative, the effect of this is moot
            layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer]) # 对hidden states按照之前用relative hidden states得到的均值进行recenter

            # get the signs for each component
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):

                transformed_hidden_states = project_onto_direction(layer_hidden_states, self.directions[layer][component_index]) # 在第一主成分方向进行投影，shape: (batch,)
                
                pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))] # 对于每个训练标签，提取与之对应的投影后的分数。

                # We do elements instead of argmin/max because sometimes we pad random choices in training
                pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)]) # 具体来说，它遍历每个pca_outputs_comp（即转换后的隐藏状态），并检查每个状态中的最小值是否与标签为1的位置相匹配。这会返回一个布尔值的列表，然后计算这个列表的平均值，得到最小值的平均出现频率。
                pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])

       
                layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min)) # 如果最大值的平均出现频率大于最小值的平均出现频率，则为1【不需要改变方向】，否则为-1。这个值是一个标量，表示这个方向的正负性。
                if layer_signs[component_index] == 0: # 两个方向都可以
                    layer_signs[component_index] = 1 # default to positive in case of tie

            signs[layer] = layer_signs

        return signs
    
class ClusterMeanRepReader(RepReader):
    """Get the direction that is the difference between the mean of the positive and negative clusters."""
    n_components = 1
    needs_hiddens = True

    def __init__(self):
        super().__init__()

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):

        # train labels is necessary to differentiate between different classes
        train_choices = kwargs['train_choices'] if 'train_choices' in kwargs else None
        assert train_choices is not None, "ClusterMeanRepReader requires train_choices to differentiate two clusters"
        for layer in hidden_layers:
            assert len(train_choices) == len(hidden_states[layer]), f"Shape mismatch between hidden states ({len(hidden_states[layer])}) and labels ({len(train_choices)})"

        train_choices = np.array(train_choices)
        neg_class = np.where(train_choices == 0)
        pos_class = np.where(train_choices == 1)

        directions = {}
        for layer in hidden_layers:
            H_train = np.array(hidden_states[layer])

            H_pos_mean = H_train[pos_class].mean(axis=0, keepdims=True)
            H_neg_mean = H_train[neg_class].mean(axis=0, keepdims=True)

            directions[layer] = H_pos_mean - H_neg_mean
        
        return directions


class RandomRepReader(RepReader):
    """Get random directions for each hidden layer. Do not use hidden 
    states or train labels of any kind."""

    def __init__(self, needs_hiddens=True):
        super().__init__()

        self.n_components = 1
        self.needs_hiddens = needs_hiddens

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):

        directions = {}
        for layer in hidden_layers:
            directions[layer] = np.expand_dims(np.random.randn(model.config.hidden_size), 0)

        return directions


DIRECTION_FINDERS = {
    'pca': PCARepReader,
    'cluster_mean': ClusterMeanRepReader,
    'random': RandomRepReader,
}