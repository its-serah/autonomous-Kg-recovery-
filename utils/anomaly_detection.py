"""
Base anomaly detection for self-healing graph system.
"""
class BaseDetector:
    """Base class for all anomaly detection mechanisms."""
    
    def __init__(self, graph):
        """Initialize with a graph to analyze.
        
        Args:
            graph: A NetworkX graph or any graph-like object
        """
        self.graph = graph
        self.anomalies = {}
    
    def detect(self):
        """Detect anomalies in the graph.
        
        This is a base method that should be overridden by subclasses.
        
        Returns:
            Dictionary with detected anomalies
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def summarize(self):
        """Summarize detected anomalies.
        
        Returns:
            Dictionary summarizing detected anomalies
        """
        return self.anomalies
