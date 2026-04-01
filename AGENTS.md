# AGENTS.md - Development Guidelines for Django-medic

This file provides guidance for AI agents working in this codebase.

## Project Overview

This is a Django 4.2.17 web application for medical/health assessment. It includes:
- User management (User model)
- Health assessment tracking (Assessment model)
- SRGA (Spectral/ML health assessment module)
- Dashboard functionality

## Build/Lint/Test Commands

### Running the Django Server
```bash
python manage.py runserver
```

### Running Tests
```bash
python manage.py test
```

### Running a Single Test
```bash
python manage.py test main.tests.<TestClassName>.<test_method_name>
```

Example: `python manage.py test main.tests.TestUser.test_user_creation`

### Django System Check
```bash
python manage.py check
```

### Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Collecting Static Files
```bash
python manage.py collectstatic
```

## Code Style Guidelines

### General
- Use 4 spaces for indentation (Python standard)
- Maximum line length: 120 characters
- Use type hints for function signatures where helpful
- Avoid trailing whitespace

### Python/Django Conventions
- Follow PEP 8 style guide
- Use `from django.conf import settings` for settings access
- Use `models.Model` base class for all models
- Use `verbose_name` for all model fields (Chinese acceptable for user-facing)
- Use `db_index=True` for frequently queried fields

### Naming Conventions
- **Models**: PascalCase (e.g., `User`, `Assessment`)
- **Functions/Variables**: snake_case (e.g., `get_user_data`, `user_id`)
- **Constants**: UPPER_SNAKE_CASE
- **URL names**: snake_case with underscores
- **Template names**: lowercase with underscores

### Imports
Standard order (per PEP 8):
1. Standard library
2. Third-party
3. Django
4. Local application

Example:
```python
import os
from pathlib import Path
from typing import Optional

from django.db import models
from django.http import JsonResponse

from main.models import User
```

### Models (models.py)
- Use `db_table` to specify custom table names
- Set `managed = False` for existing database tables
- Include `__str__` method for all models
- Use `Meta` class for ordering and verbose names

### Views (views.py)
- Use Django views (function-based or class-based)
- Return proper HTTP responses (JsonResponse for APIs)
- Handle exceptions gracefully with try/except
- Return appropriate HTTP status codes (200, 400, 404, etc.)

### Templates
- Located in `templates/` directory within apps
- Use Django template syntax `{% %}` and `{{ }}`
- Follow Django template conventions

### Error Handling
- Use try/except blocks for operations that may fail
- Log errors appropriately (use Django's logging)
- Return user-friendly error messages
- For API endpoints, return JSON with error details

### Database
- Use Django ORM for all database operations
- Avoid raw SQL unless necessary
- Use migrations for schema changes
- Use `.objects.filter()` for queries

### Configuration
- All configuration via environment variables in `.env`
- Use `os.environ.get()` with sensible defaults
- Settings in `config/settings.py`

### File Path Management
All AI-generated files must follow these directory conventions:

- **Code/Script files** (.py, .js, etc.): `./main/agent/temp/`
- **Result files** (ppt, doc, pdf, xls, etc.): `./main/agent/output/`

Example:
```python
temp_dir = BASE_DIR / "main" / "agent" / "temp"
output_dir = BASE_DIR / "main" / "agent" / "output"
```

## Common Patterns

### API View Pattern
```python
def api_view(request):
    try:
        data = process_request(request)
        return JsonResponse({"ok": True, "data": data})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)
```

### Model Query Pattern
```python
users = User.objects.filter(userID=user_id).order_by('-created_at')
```

## Directory Structure
```
Django-medic/
├── config/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── main/            # Main Django app
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── admin.py
│   ├── apps.py
│   ├── tests.py
│   ├── data/        # Data processing modules
│   └── SRGA/        # ML/health assessment module
├── manage.py
└── requirements.txt
```

## Dependencies
- Django==4.2.17
- pymysql>=1.1.0
- python-dotenv>=1.0.0
- pillow>=11.0.0
- pyyaml>=6.0.0

## Notes for AI Agents
- This project uses Chinese (Simplified) for user-facing strings
- The project connects to a MySQL database named `web_medic`
- SRGA module requires PyTorch/torchvision/torchaudio
- USE_TZ is set to False in settings