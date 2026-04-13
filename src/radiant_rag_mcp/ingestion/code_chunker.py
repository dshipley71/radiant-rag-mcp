"""
Code-aware chunking for source code files.

Provides intelligent chunking that respects code structure:
- Functions, classes, methods
- Import statements
- Docstrings and comments
- Logical code blocks
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    SQL = "sql"
    SHELL = "shell"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


# File extension to language mapping
EXTENSION_TO_LANGUAGE = {
    # Python
    ".py": CodeLanguage.PYTHON,
    ".pyw": CodeLanguage.PYTHON,
    ".pyx": CodeLanguage.PYTHON,
    # JavaScript/TypeScript
    ".js": CodeLanguage.JAVASCRIPT,
    ".jsx": CodeLanguage.JAVASCRIPT,
    ".mjs": CodeLanguage.JAVASCRIPT,
    ".ts": CodeLanguage.TYPESCRIPT,
    ".tsx": CodeLanguage.TYPESCRIPT,
    # Java/JVM
    ".java": CodeLanguage.JAVA,
    ".kt": CodeLanguage.KOTLIN,
    ".kts": CodeLanguage.KOTLIN,
    ".scala": CodeLanguage.SCALA,
    # Systems
    ".go": CodeLanguage.GO,
    ".rs": CodeLanguage.RUST,
    ".c": CodeLanguage.C,
    ".h": CodeLanguage.C,
    ".cpp": CodeLanguage.CPP,
    ".cc": CodeLanguage.CPP,
    ".cxx": CodeLanguage.CPP,
    ".hpp": CodeLanguage.CPP,
    ".hxx": CodeLanguage.CPP,
    ".cs": CodeLanguage.CSHARP,
    # Scripting
    ".rb": CodeLanguage.RUBY,
    ".php": CodeLanguage.PHP,
    ".swift": CodeLanguage.SWIFT,
    ".r": CodeLanguage.R,
    ".R": CodeLanguage.R,
    # Shell
    ".sh": CodeLanguage.SHELL,
    ".bash": CodeLanguage.SHELL,
    ".zsh": CodeLanguage.SHELL,
    # Data
    ".sql": CodeLanguage.SQL,
    ".yaml": CodeLanguage.YAML,
    ".yml": CodeLanguage.YAML,
    ".json": CodeLanguage.JSON,
    # Documentation
    ".md": CodeLanguage.MARKDOWN,
    ".mdx": CodeLanguage.MARKDOWN,
    ".rst": CodeLanguage.MARKDOWN,
    ".txt": CodeLanguage.MARKDOWN,
}


@dataclass
class CodeBlock:
    """Represents a logical block of code."""
    
    content: str
    block_type: str  # "class", "function", "method", "imports", "module_doc", "code"
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    language: CodeLanguage = CodeLanguage.UNKNOWN
    parent: Optional[str] = None  # For methods, the class name
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        """Get full qualified name."""
        if self.parent and self.name:
            return f"{self.parent}.{self.name}"
        return self.name or self.block_type


@dataclass 
class CodeChunk:
    """A chunk of code ready for indexing."""
    
    content: str
    language: CodeLanguage
    file_path: str
    block_type: str
    block_name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    imports_context: Optional[str] = None  # Relevant imports for this chunk
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_indexable_text(self) -> str:
        """Convert to text suitable for indexing."""
        parts = []
        
        # Add context header
        header = f"File: {self.file_path}"
        if self.block_name:
            header += f" | {self.block_type}: {self.block_name}"
        header += f" | Language: {self.language.value}"
        parts.append(header)
        
        # Add imports context if available
        if self.imports_context:
            parts.append(f"Imports:\n{self.imports_context}")
        
        # Add the code
        parts.append(f"Code:\n{self.content}")
        
        return "\n\n".join(parts)


class CodeParser:
    """
    Language-aware code parser.
    
    Extracts logical blocks (functions, classes, etc.) from source code.
    """
    
    # Python patterns
    PYTHON_CLASS = re.compile(
        r'^(class\s+(\w+).*?:.*?)(?=^class\s|\Z)',
        re.MULTILINE | re.DOTALL
    )
    PYTHON_FUNCTION = re.compile(
        r'^((?:async\s+)?def\s+(\w+)\s*\([^)]*\).*?:.*?)(?=^(?:async\s+)?def\s|^class\s|\Z)',
        re.MULTILINE | re.DOTALL
    )
    PYTHON_IMPORTS = re.compile(
        r'^((?:from\s+\S+\s+)?import\s+.+)$',
        re.MULTILINE
    )
    PYTHON_DOCSTRING = re.compile(
        r'^("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
        re.MULTILINE
    )
    
    # JavaScript/TypeScript patterns
    JS_CLASS = re.compile(
        r'^((?:export\s+)?class\s+(\w+).*?\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    JS_FUNCTION = re.compile(
        r'^((?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    JS_ARROW_FUNCTION = re.compile(
        r'^((?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*(?:\{[\s\S]*?\n\}|[^\n]+))',
        re.MULTILINE
    )
    JS_IMPORTS = re.compile(
        r'^(import\s+.*?(?:from\s+[\'"][^\'"]+[\'"])?;?)$',
        re.MULTILINE
    )
    
    # Java patterns
    JAVA_CLASS = re.compile(
        r'^((?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:abstract\s+)?class\s+(\w+).*?\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    JAVA_METHOD = re.compile(
        r'^(\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{[\s\S]*?\n\s*\})',
        re.MULTILINE
    )
    JAVA_IMPORTS = re.compile(
        r'^(import\s+(?:static\s+)?[\w.]+(?:\.\*)?;)$',
        re.MULTILINE
    )
    
    # Go patterns
    GO_FUNCTION = re.compile(
        r'^(func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\).*?\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    GO_TYPE = re.compile(
        r'^(type\s+(\w+)\s+(?:struct|interface)\s*\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    GO_IMPORTS = re.compile(
        r'^(import\s+(?:\(\s*[\s\S]*?\s*\)|"[^"]+"))$',
        re.MULTILINE
    )
    
    # Rust patterns
    RUST_FUNCTION = re.compile(
        r'^((?:pub\s+)?(?:async\s+)?fn\s+(\w+).*?\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    RUST_STRUCT = re.compile(
        r'^((?:pub\s+)?struct\s+(\w+).*?\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    RUST_IMPL = re.compile(
        r'^(impl(?:<[^>]+>)?\s+(?:\w+\s+for\s+)?(\w+).*?\{[\s\S]*?\n\})',
        re.MULTILINE
    )
    RUST_USE = re.compile(
        r'^(use\s+[\w:]+(?:::\{[^}]+\})?;)$',
        re.MULTILINE
    )
    
    def __init__(self):
        self._parsers = {
            CodeLanguage.PYTHON: self._parse_python,
            CodeLanguage.JAVASCRIPT: self._parse_javascript,
            CodeLanguage.TYPESCRIPT: self._parse_javascript,  # Similar enough
            CodeLanguage.JAVA: self._parse_java,
            CodeLanguage.GO: self._parse_go,
            CodeLanguage.RUST: self._parse_rust,
        }
    
    def detect_language(self, filename: str) -> CodeLanguage:
        """Detect language from filename."""
        import os
        ext = os.path.splitext(filename)[1].lower()
        return EXTENSION_TO_LANGUAGE.get(ext, CodeLanguage.UNKNOWN)
    
    def parse(self, content: str, filename: str) -> List[CodeBlock]:
        """
        Parse source code into logical blocks.
        
        Args:
            content: Source code content
            filename: Filename for language detection
            
        Returns:
            List of CodeBlock objects
        """
        language = self.detect_language(filename)
        
        if language in self._parsers:
            return self._parsers[language](content, language)
        else:
            # Generic fallback - treat whole file as one block
            return [CodeBlock(
                content=content,
                block_type="file",
                name=filename,
                language=language,
            )]
    
    def _parse_python(self, content: str, language: CodeLanguage) -> List[CodeBlock]:
        """Parse Python source code."""
        blocks = []
        lines = content.split('\n')
        
        # Extract imports
        imports = self.PYTHON_IMPORTS.findall(content)
        if imports:
            import_block = '\n'.join(imports)
            blocks.append(CodeBlock(
                content=import_block,
                block_type="imports",
                language=language,
            ))
        
        # Extract module docstring
        if content.strip().startswith(('"""', "'''")):
            match = self.PYTHON_DOCSTRING.match(content.strip())
            if match:
                blocks.append(CodeBlock(
                    content=match.group(1),
                    block_type="module_doc",
                    language=language,
                ))
        
        # Use AST for better Python parsing
        try:
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 50
                    class_content = '\n'.join(lines[start:end])
                    
                    docstring = ast.get_docstring(node)
                    
                    blocks.append(CodeBlock(
                        content=class_content,
                        block_type="class",
                        name=node.name,
                        start_line=start,
                        end_line=end,
                        language=language,
                        docstring=docstring,
                    ))
                    
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Skip methods (they're inside classes)
                    parent_class = None
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            for child in ast.iter_child_nodes(parent):
                                if child is node:
                                    parent_class = parent.name
                                    break
                    
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 30
                    func_content = '\n'.join(lines[start:end])
                    
                    docstring = ast.get_docstring(node)
                    
                    blocks.append(CodeBlock(
                        content=func_content,
                        block_type="method" if parent_class else "function",
                        name=node.name,
                        start_line=start,
                        end_line=end,
                        language=language,
                        parent=parent_class,
                        docstring=docstring,
                    ))
                    
        except SyntaxError:
            # Fall back to regex parsing
            blocks.extend(self._parse_python_regex(content, language))
        
        return blocks if blocks else [CodeBlock(
            content=content,
            block_type="file",
            language=language,
        )]
    
    def _parse_python_regex(self, content: str, language: CodeLanguage) -> List[CodeBlock]:
        """Fallback regex-based Python parsing."""
        blocks = []
        
        # Find classes
        for match in self.PYTHON_CLASS.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="class",
                name=match.group(2),
                language=language,
            ))
        
        # Find functions (outside classes)
        for match in self.PYTHON_FUNCTION.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="function",
                name=match.group(2),
                language=language,
            ))
        
        return blocks
    
    def _parse_javascript(self, content: str, language: CodeLanguage) -> List[CodeBlock]:
        """Parse JavaScript/TypeScript source code."""
        blocks = []
        
        # Extract imports
        imports = self.JS_IMPORTS.findall(content)
        if imports:
            import_block = '\n'.join(imports)
            blocks.append(CodeBlock(
                content=import_block,
                block_type="imports",
                language=language,
            ))
        
        # Find classes
        for match in self.JS_CLASS.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="class",
                name=match.group(2),
                language=language,
            ))
        
        # Find functions
        for match in self.JS_FUNCTION.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="function",
                name=match.group(2),
                language=language,
            ))
        
        # Find arrow functions
        for match in self.JS_ARROW_FUNCTION.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="function",
                name=match.group(2),
                language=language,
            ))
        
        return blocks if blocks else [CodeBlock(
            content=content,
            block_type="file",
            language=language,
        )]
    
    def _parse_java(self, content: str, language: CodeLanguage) -> List[CodeBlock]:
        """Parse Java source code."""
        blocks = []
        
        # Extract imports
        imports = self.JAVA_IMPORTS.findall(content)
        if imports:
            import_block = '\n'.join(imports)
            blocks.append(CodeBlock(
                content=import_block,
                block_type="imports",
                language=language,
            ))
        
        # Find classes
        for match in self.JAVA_CLASS.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="class",
                name=match.group(2),
                language=language,
            ))
        
        # Find methods (standalone parsing - may overlap with class content)
        for match in self.JAVA_METHOD.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="method",
                name=match.group(2),
                language=language,
            ))
        
        return blocks if blocks else [CodeBlock(
            content=content,
            block_type="file",
            language=language,
        )]
    
    def _parse_go(self, content: str, language: CodeLanguage) -> List[CodeBlock]:
        """Parse Go source code."""
        blocks = []
        
        # Extract imports
        imports = self.GO_IMPORTS.findall(content)
        if imports:
            import_block = '\n'.join(imports)
            blocks.append(CodeBlock(
                content=import_block,
                block_type="imports",
                language=language,
            ))
        
        # Find types (structs/interfaces)
        for match in self.GO_TYPE.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="type",
                name=match.group(2),
                language=language,
            ))
        
        # Find functions
        for match in self.GO_FUNCTION.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="function",
                name=match.group(2),
                language=language,
            ))
        
        return blocks if blocks else [CodeBlock(
            content=content,
            block_type="file",
            language=language,
        )]
    
    def _parse_rust(self, content: str, language: CodeLanguage) -> List[CodeBlock]:
        """Parse Rust source code."""
        blocks = []
        
        # Extract use statements
        uses = self.RUST_USE.findall(content)
        if uses:
            use_block = '\n'.join(uses)
            blocks.append(CodeBlock(
                content=use_block,
                block_type="imports",
                language=language,
            ))
        
        # Find structs
        for match in self.RUST_STRUCT.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="struct",
                name=match.group(2),
                language=language,
            ))
        
        # Find impl blocks
        for match in self.RUST_IMPL.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="impl",
                name=match.group(2),
                language=language,
            ))
        
        # Find functions
        for match in self.RUST_FUNCTION.finditer(content):
            blocks.append(CodeBlock(
                content=match.group(1),
                block_type="function",
                name=match.group(2),
                language=language,
            ))
        
        return blocks if blocks else [CodeBlock(
            content=content,
            block_type="file",
            language=language,
        )]


class CodeChunker:
    """
    Intelligent code chunker.
    
    Converts parsed code blocks into chunks suitable for indexing.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        include_imports_context: bool = True,
    ):
        """
        Initialize chunker.
        
        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size (smaller blocks are combined)
            include_imports_context: Include relevant imports in chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_imports_context = include_imports_context
        self.parser = CodeParser()
    
    def chunk_file(
        self,
        content: str,
        file_path: str,
    ) -> List[CodeChunk]:
        """
        Chunk a source code file.
        
        Args:
            content: File content
            file_path: File path for language detection
            
        Returns:
            List of CodeChunk objects
        """
        # Parse into blocks
        blocks = self.parser.parse(content, file_path)
        
        if not blocks:
            return []
        
        # Extract imports for context
        imports_context = None
        if self.include_imports_context:
            import_blocks = [b for b in blocks if b.block_type == "imports"]
            if import_blocks:
                imports_context = import_blocks[0].content
        
        # Convert blocks to chunks
        chunks = []
        small_blocks = []  # Buffer for combining small blocks
        
        for block in blocks:
            # Skip standalone import blocks (they're added as context)
            if block.block_type == "imports":
                continue
            
            if len(block.content) > self.max_chunk_size:
                # Large block - split it
                chunks.extend(self._split_large_block(
                    block, file_path, imports_context
                ))
            elif len(block.content) < self.min_chunk_size:
                # Small block - buffer for combining
                small_blocks.append(block)
            else:
                # Normal block - create chunk
                chunks.append(self._block_to_chunk(
                    block, file_path, imports_context
                ))
        
        # Combine buffered small blocks
        if small_blocks:
            chunks.extend(self._combine_small_blocks(
                small_blocks, file_path, imports_context
            ))
        
        return chunks
    
    def _block_to_chunk(
        self,
        block: CodeBlock,
        file_path: str,
        imports_context: Optional[str],
    ) -> CodeChunk:
        """Convert a CodeBlock to a CodeChunk."""
        return CodeChunk(
            content=block.content,
            language=block.language,
            file_path=file_path,
            block_type=block.block_type,
            block_name=block.full_name,
            start_line=block.start_line,
            end_line=block.end_line,
            imports_context=imports_context,
            metadata={
                "docstring": block.docstring,
                "parent": block.parent,
            },
        )
    
    def _split_large_block(
        self,
        block: CodeBlock,
        file_path: str,
        imports_context: Optional[str],
    ) -> List[CodeChunk]:
        """Split a large block into smaller chunks."""
        chunks = []
        content = block.content
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Create chunk from buffer
                chunk_content = '\n'.join(current_chunk)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    language=block.language,
                    file_path=file_path,
                    block_type=block.block_type,
                    block_name=f"{block.full_name}_part{chunk_index}" if block.name else None,
                    imports_context=imports_context,
                    metadata={"part": chunk_index, "parent_block": block.name},
                ))
                current_chunk = []
                current_size = 0
                chunk_index += 1
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(CodeChunk(
                content=chunk_content,
                language=block.language,
                file_path=file_path,
                block_type=block.block_type,
                block_name=f"{block.full_name}_part{chunk_index}" if block.name else None,
                imports_context=imports_context,
                metadata={"part": chunk_index, "parent_block": block.name},
            ))
        
        return chunks
    
    def _combine_small_blocks(
        self,
        blocks: List[CodeBlock],
        file_path: str,
        imports_context: Optional[str],
    ) -> List[CodeChunk]:
        """Combine small blocks into larger chunks."""
        if not blocks:
            return []
        
        chunks = []
        current_blocks = []
        current_size = 0
        
        for block in blocks:
            block_size = len(block.content)
            
            if current_size + block_size > self.max_chunk_size and current_blocks:
                # Create combined chunk
                combined_content = '\n\n'.join(b.content for b in current_blocks)
                block_names = [b.name for b in current_blocks if b.name]
                
                chunks.append(CodeChunk(
                    content=combined_content,
                    language=current_blocks[0].language,
                    file_path=file_path,
                    block_type="combined",
                    block_name=", ".join(block_names) if block_names else None,
                    imports_context=imports_context,
                    metadata={"combined_blocks": len(current_blocks)},
                ))
                current_blocks = []
                current_size = 0
            
            current_blocks.append(block)
            current_size += block_size
        
        # Add remaining blocks
        if current_blocks:
            combined_content = '\n\n'.join(b.content for b in current_blocks)
            block_names = [b.name for b in current_blocks if b.name]
            
            chunks.append(CodeChunk(
                content=combined_content,
                language=current_blocks[0].language,
                file_path=file_path,
                block_type="combined",
                block_name=", ".join(block_names) if block_names else None,
                imports_context=imports_context,
                metadata={"combined_blocks": len(current_blocks)},
            ))
        
        return chunks


def chunk_code_file(
    content: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> List[CodeChunk]:
    """
    Convenience function to chunk a code file.
    
    Args:
        content: File content
        file_path: File path
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of CodeChunk objects
    """
    chunker = CodeChunker(max_chunk_size=max_chunk_size)
    return chunker.chunk_file(content, file_path)
