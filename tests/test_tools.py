from unittest.mock import patch, mock_open
from ai_cli.tools.read_file import ReadFileTool
from ai_cli.tools.edit_file import EditFileTool

# --- Tests for ReadFileTool ---

def test_read_file_success():
    """Test reading a file that exists."""
    tool = ReadFileTool()
    mock_content = "def hello_world():\n    print('hello')"
    
    # We "mock" the open() function so it returns our fake content instead of reading a real file
    with patch("builtins.open", mock_open(read_data=mock_content)):
        result = tool.execute("dummy_path.py")
        
    assert "File contents of dummy_path.py:" in result
    assert "def hello_world():" in result

def test_read_file_not_found():
    """Test reading a file that does not exist."""
    tool = ReadFileTool()
    
    # We force the open() function to raise a FileNotFoundError
    with patch("builtins.open", side_effect=FileNotFoundError):
        result = tool.execute("missing_file.py")
        
    assert result == "File not found: missing_file.py"

# --- Tests for EditFileTool ---

def test_edit_file_create_new():
    """Test creating a new file (when old_text is empty)."""
    tool = EditFileTool()
    
    # Mocking os.path.exists to return False (simulating the file doesn't exist)
    # Mocking input to automatically return 'y' (simulating user confirmation)
    with patch("os.path.exists", return_value=False), \
         patch("builtins.input", return_value="y"), \
         patch("builtins.open", mock_open()) as mock_file:
         
        result = tool.execute(path="new_script.py", new_text="print('New File')", old_text="")
        
    assert result == "Successfully created new_script.py"
    # Verify the file was opened in write mode ('w') and the text was written
    mock_file.assert_called_with("new_script.py", "w", encoding="utf-8")
    mock_file().write.assert_called_once_with("print('New File')")

def test_edit_file_blocked_by_user():
    """Test that the tool aborts if the user types 'n' at the confirmation prompt."""
    tool = EditFileTool()
    
    with patch("builtins.input", return_value="n"):
        result = tool.execute(path="critical_system.py", new_text="bad code", old_text="")
        
    assert result == "Operation blocked: User denied permission to edit critical_system.py."
