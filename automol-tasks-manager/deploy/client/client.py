"""
Python client for AutoMol prediction server.

Provides easy-to-use interface for making predictions via REST API.

Example:
    >>> from deploy.client import AutoMolClient
    >>> client = AutoMolClient("http://localhost:8000")
    >>> result = client.predict("CCO")
    >>> print(result)
"""

import json
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None


class AutoMolClient:
    """
    Client for AutoMol prediction server.

    Provides methods for:
    - Single and batch predictions
    - Feature extraction
    - Health checks
    - Model information

    Example:
        >>> client = AutoMolClient("http://localhost:8000")
        >>> client.health()
        {'status': 'healthy', 'model_loaded': True, ...}

        >>> client.predict("CCO")
        {'pIC50': 5.23, 'LogP': 1.2}

        >>> client.predict_batch(["CCO", "CCN"])
        [{'pIC50': 5.23, 'LogP': 1.2}, {'pIC50': 6.15, 'LogP': 1.5}]
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 60,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            base_url: Server URL (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        if requests is None:
            raise ImportError(
                "requests library required for client. "
                "Install with: pip install requests"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

        # Session for connection pooling
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Make HTTP request to server."""
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to server at {self.base_url}. "
                "Is the server running?"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request timed out after {self.timeout}s. "
                "Try increasing timeout or reducing batch size."
            )
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("detail", "")
            except Exception:
                pass
            raise RuntimeError(f"Server error: {e}. {error_detail}")

    def health(self) -> Dict[str, Any]:
        """
        Check server health.

        Returns:
            Health status dictionary
        """
        return self._request("GET", "/health")

    def is_healthy(self) -> bool:
        """
        Check if server is healthy and model is loaded.

        Returns:
            True if server is ready for predictions
        """
        try:
            health = self.health()
            return health.get("status") == "healthy" and health.get("model_loaded", False)
        except Exception:
            return False

    def model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Model metadata dictionary
        """
        return self._request("GET", "/model/info")

    def get_properties(self) -> List[str]:
        """
        Get available prediction properties.

        Returns:
            List of property names
        """
        info = self.model_info()
        return info.get("properties", [])

    def predict(
        self,
        smiles: str,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Predict for a single SMILES.

        Args:
            smiles: SMILES string
            properties: List of properties to predict (None for all)

        Returns:
            Dictionary of property -> predicted value
        """
        data = {"smiles": smiles}
        if properties:
            data["properties"] = properties

        response = self._request("POST", "/predict", data=data)

        if response.get("error"):
            raise ValueError(f"Prediction error: {response['error']}")

        return response.get("predictions", {})

    def predict_batch(
        self,
        smiles_list: List[str],
        properties: Optional[List[str]] = None,
        return_errors: bool = False,
    ) -> Union[List[Dict[str, Optional[float]]], Dict[str, Any]]:
        """
        Predict for multiple SMILES.

        Args:
            smiles_list: List of SMILES strings
            properties: List of properties to predict (None for all)
            return_errors: If True, return full response with errors

        Returns:
            List of prediction dictionaries (or full response if return_errors=True)
        """
        data = {"smiles_list": smiles_list}
        if properties:
            data["properties"] = properties

        response = self._request("POST", "/predict/batch", data=data)

        if return_errors:
            return response

        return [p["predictions"] for p in response["predictions"]]

    def predict_df(
        self,
        df,
        smiles_column: str = "SMILES",
        properties: Optional[List[str]] = None,
        batch_size: int = 100,
    ):
        """
        Add predictions to a DataFrame.

        Args:
            df: pandas DataFrame with SMILES column
            smiles_column: Name of SMILES column
            properties: Properties to predict (None for all)
            batch_size: Batch size for predictions

        Returns:
            DataFrame with prediction columns added
        """
        if pd is None:
            raise ImportError("pandas required for predict_df")

        result = df.copy()
        smiles_list = df[smiles_column].tolist()

        # Get available properties if not specified
        if properties is None:
            properties = self.get_properties()

        # Initialize prediction columns
        for prop in properties:
            result[f"predicted_{prop}"] = None

        # Predict in batches
        all_predictions = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            predictions = self.predict_batch(batch, properties=properties)
            all_predictions.extend(predictions)

        # Fill in predictions
        for i, pred in enumerate(all_predictions):
            for prop, value in pred.items():
                result.at[i, f"predicted_{prop}"] = value

        return result

    def predict_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        smiles_column: str = "SMILES",
        properties: Optional[List[str]] = None,
        batch_size: int = 100,
    ):
        """
        Predict for SMILES in a CSV file.

        Args:
            input_path: Path to input CSV
            output_path: Path for output CSV (optional)
            smiles_column: Name of SMILES column
            properties: Properties to predict (None for all)
            batch_size: Batch size for predictions

        Returns:
            DataFrame with predictions
        """
        if pd is None:
            raise ImportError("pandas required for predict_file")

        df = pd.read_csv(input_path)
        result = self.predict_df(
            df,
            smiles_column=smiles_column,
            properties=properties,
            batch_size=batch_size,
        )

        if output_path:
            result.to_csv(output_path, index=False)

        return result

    def get_features(
        self,
        smiles_list: List[str],
    ) -> Dict[str, Any]:
        """
        Get feature vectors for SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary with feature names -> feature arrays
        """
        return self._request(
            "POST",
            "/features",
            data={"smiles_list": smiles_list}
        )

    def wait_for_server(
        self,
        timeout: int = 120,
        interval: int = 2,
    ) -> bool:
        """
        Wait for server to become available.

        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            True if server became available, False if timeout
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            if self.is_healthy():
                return True
            time.sleep(interval)

        return False

    def __repr__(self) -> str:
        return f"AutoMolClient(base_url='{self.base_url}')"


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI for making predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoMol Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--smiles", type=str, help="Single SMILES to predict")
    parser.add_argument("--file", type=str, help="CSV file with SMILES")
    parser.add_argument("--output", type=str, help="Output CSV file")
    parser.add_argument("--column", default="SMILES", help="SMILES column name")
    parser.add_argument("--properties", type=str, nargs="+", help="Properties to predict")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--info", action="store_true", help="Get model info")

    args = parser.parse_args()

    client = AutoMolClient(args.server)

    if args.health:
        print(json.dumps(client.health(), indent=2))
        return

    if args.info:
        print(json.dumps(client.model_info(), indent=2))
        return

    if args.smiles:
        result = client.predict(args.smiles, properties=args.properties)
        print(f"{args.smiles}:")
        for prop, value in result.items():
            print(f"  {prop}: {value}")
        return

    if args.file:
        result = client.predict_file(
            args.file,
            output_path=args.output,
            smiles_column=args.column,
            properties=args.properties,
            batch_size=args.batch_size,
        )
        if args.output:
            print(f"Results saved to {args.output}")
        else:
            print(result.to_string())
        return

    parser.print_help()


if __name__ == "__main__":
    main()
