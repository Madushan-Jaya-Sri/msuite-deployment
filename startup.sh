#!/bin/bash
python manage.py collectstatic && gunicorn --workers 2 m_suite_project.wsgi 