"""Utility functions for logging module"""

import os


def path_to_slug(path: str) -> str:
    """
    Convert file path to slug suitable for directory names
    
    Examples:
        /Users/mingmu/projects/app -> Users-mingmu-projects-app
        /home/user/my project -> home-user-my_project
    """
    abs_path = os.path.abspath(path)
    
    # Remove leading slash
    if abs_path.startswith('/'):
        abs_path = abs_path[1:]
    
    # Replace slashes and spaces
    slug = abs_path.replace('/', '-').replace(' ', '_')
    
    # Handle Windows drive letters (C: -> C-)
    if os.name == 'nt' and ':' in slug:
        slug = slug.replace(':', '-')
    
    return slug

