import pytest
from src.main import main


def test_main_function():
    """Test that main function runs without error."""
    # This is a basic test - you can expand this as needed
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"main() raised {e} unexpectedly!")
