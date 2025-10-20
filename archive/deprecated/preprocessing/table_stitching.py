"""
Table Stitching Engine - Uses Pandas & NumPy for multi-page table combining.
Handles tables that span multiple pages using coordinate-based alignment.
Based on recommendation: Pandas & NumPy for aligning columns by approximate x-coordinate.
"""

import logging
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TableStitchingEngine:
    """
    Stitches tables that span multiple pages using Pandas & NumPy.
    Combines table fragments into complete table structures using coordinate alignment.
    Follows recommended approach: align columns by approximate x-coordinate.
    """

    def __init__(self):
        """Initialize table stitching engine with Pandas & NumPy."""
        self.initialized = True
        logger.info("TableStitchingEngine initialized with Pandas & NumPy")

    def stitch_tables(self, tables: List[Dict]) -> Dict[str, Any]:
        """
        Stitch table fragments across pages using coordinate-based alignment.

        Args:
            tables: List of detected tables from table detection

        Returns:
            Dictionary with stitched tables using Pandas/NumPy alignment
        """
        if not tables:
            return {"stitched_tables": [], "method": "none"}

        # Group tables by proximity and column structure similarity
        table_groups = self._group_tables_by_coordinates(tables)

        stitched_tables = []
        for group in table_groups:
            if len(group) > 1:
                # Multiple tables to stitch using coordinate alignment
                stitched = self._stitch_with_pandas_numpy(group)
                stitched_tables.append(stitched)
            else:
                # Single table, no stitching needed
                stitched_tables.append(group[0])

        return {
            "stitched_tables": stitched_tables,
            "original_count": len(tables),
            "stitched_count": len(stitched_tables),
            "method": "pandas_numpy_coordinate_alignment",
        }

    def _group_tables_by_coordinates(self, tables: List[Dict]) -> List[List[Dict]]:
        """
        Group tables using coordinate-based analysis with NumPy.

        Args:
            tables: List of detected tables

        Returns:
            List of table groups based on coordinate alignment
        """
        if not tables:
            return []

        # Convert to numpy arrays for coordinate analysis
        table_data = []
        for i, table in enumerate(tables):
            bbox = table.get("bbox", [0, 0, 0, 0])
            table_data.append(
                {
                    "index": i,
                    "page": table.get("page", 0),
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                    "center_x": (bbox[0] + bbox[2]) / 2,
                    "table": table,
                }
            )

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(table_data)

        # Group by approximate x-coordinate alignment (column alignment)
        groups = []
        tolerance = 50  # Pixel tolerance for alignment

        remaining_indices = set(df.index)

        while remaining_indices:
            # Start new group with first remaining table
            seed_idx = next(iter(remaining_indices))
            seed_table = df.loc[seed_idx]
            group_indices = {seed_idx}
            remaining_indices.remove(seed_idx)

            # Find tables with similar x-coordinates (column alignment)
            for idx in list(remaining_indices):
                table_row = df.loc[idx]

                # Check if tables have similar column structure
                if self._tables_have_similar_columns(seed_table, table_row, tolerance):
                    group_indices.add(idx)
                    remaining_indices.remove(idx)

            # Convert indices back to table objects
            group = [
                df.loc[idx]["table"]
                for idx in sorted(group_indices, key=lambda x: df.loc[x]["page"])
            ]
            groups.append(group)

        return groups

    def _tables_have_similar_columns(
        self, table1: pd.Series, table2: pd.Series, tolerance: float
    ) -> bool:
        """
        Check if two tables have similar column structure using coordinate analysis.

        Args:
            table1: First table data
            table2: Second table data
            tolerance: Coordinate tolerance in pixels

        Returns:
            True if tables should be grouped together
        """
        # Must be on consecutive or nearby pages
        page_diff = abs(table2["page"] - table1["page"])
        if page_diff > 2:
            return False

        # Check x-coordinate alignment (similar column positions)
        x_center_diff = abs(table2["center_x"] - table1["center_x"])
        width_diff = abs(table2["width"] - table1["width"])

        # Tables should have similar horizontal positioning and width
        if x_center_diff > tolerance or width_diff > tolerance:
            return False

        return True

    def _stitch_with_pandas_numpy(self, table_group: List[Dict]) -> Dict[str, Any]:
        """
        Stitch tables using Pandas DataFrame operations and NumPy coordinate alignment.

        Args:
            table_group: List of tables to stitch

        Returns:
            Stitched table with combined data
        """
        if not table_group:
            return {}

        if len(table_group) == 1:
            return table_group[0]

        # Sort tables by page number for proper stitching order
        sorted_tables = sorted(table_group, key=lambda t: t.get("page", 0))

        # Extract and combine HTML tables using pandas
        combined_df = self._combine_html_with_pandas(sorted_tables)

        # Combine bounding boxes using NumPy
        combined_bbox = self._combine_bboxes_numpy(sorted_tables)

        # Get page range
        pages = [table.get("page", 0) for table in sorted_tables]

        # Convert back to HTML
        combined_html = self._dataframe_to_html(combined_df)

        return {
            "bbox": combined_bbox,
            "pages": sorted(pages),
            "page_range": f"{min(pages)}-{max(pages)}",
            "confidence": np.mean(
                [table.get("confidence", 0.5) for table in sorted_tables]
            ),
            "html": combined_html,
            "type": "stitched_table_pandas",
            "original_count": len(sorted_tables),
            "method": "pandas_numpy_alignment",
        }

    def _combine_html_with_pandas(self, table_group: List[Dict]) -> pd.DataFrame:
        """
        Combine HTML tables using pandas for proper data alignment.

        Args:
            table_group: List of tables with HTML content

        Returns:
            Combined pandas DataFrame
        """
        dataframes = []

        for table in table_group:
            html = table.get("html", "")
            try:
                # Parse HTML table with pandas
                dfs = pd.read_html(html)
                if dfs:
                    df = dfs[0]  # Take first table from HTML
                    dataframes.append(df)
            except Exception as e:
                logger.debug(f"Failed to parse HTML table with pandas: {e}")
                # Fallback: create DataFrame from simple parsing
                df = self._simple_html_to_dataframe(html)
                if not df.empty:
                    dataframes.append(df)

        if not dataframes:
            return pd.DataFrame()

        # Combine DataFrames (concatenate rows, align columns)
        try:
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            return combined_df
        except Exception as e:
            logger.warning(f"Failed to combine DataFrames: {e}")
            return dataframes[0] if dataframes else pd.DataFrame()

    def _simple_html_to_dataframe(self, html: str) -> pd.DataFrame:
        """
        Simple HTML to DataFrame conversion as fallback.

        Args:
            html: HTML table string

        Returns:
            pandas DataFrame
        """
        # Very simple HTML parsing - extract text between tags

        # Extract table rows
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)

        data = []
        for row in rows:
            # Extract cells
            cells = re.findall(
                r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE
            )
            # Clean cell content
            clean_cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
            if clean_cells:
                data.append(clean_cells)

        if not data:
            return pd.DataFrame()

        # Create DataFrame, using first row as headers if it looks like headers
        if len(data) > 1:
            return pd.DataFrame(data[1:], columns=data[0])
        else:
            return pd.DataFrame(data)

    def _combine_bboxes_numpy(self, table_group: List[Dict]) -> List[float]:
        """
        Combine bounding boxes using NumPy for efficient coordinate operations.

        Args:
            table_group: List of tables with bbox data

        Returns:
            Combined bounding box [x0, y0, x1, y1]
        """
        if not table_group:
            return [0, 0, 0, 0]

        # Extract all bboxes and convert to numpy array
        bboxes = np.array([table.get("bbox", [0, 0, 0, 0]) for table in table_group])

        # Calculate overall bounding box using numpy operations
        x0 = np.min(bboxes[:, 0])
        y0 = np.min(bboxes[:, 1])
        x1 = np.max(bboxes[:, 2])
        y1 = np.max(bboxes[:, 3])

        return [float(x0), float(y0), float(x1), float(y1)]

    def _dataframe_to_html(self, df: pd.DataFrame) -> str:
        """
        Convert pandas DataFrame back to HTML table.

        Args:
            df: pandas DataFrame

        Returns:
            HTML table string
        """
        if df.empty:
            return "<table><tr><td>No data</td></tr></table>"

        return df.to_html(index=False, classes="stitched-table")
