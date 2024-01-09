import os
from .settings import *
from .settings import BASE_DIR


SECRET_KEY = os.environ['SECRET']


ALLOWED_HOSTS = [os.environ['WEBSITE_HOSTNAME']]

CSRF_TRUSTED_ORIGINS = ['https://'+os.environ['WEBSITE_HOSTNAME']]

DEBUG = True

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')


conn_str = os.environ['AZURE_POSTGRESQL_CONNECTIONSTRING']



#dbname=momentrosuite-database host=momentrosuite-server.postgres.database.azure.com port=5432 sslmode=require user=wfdfblpion password=N78SRBM56II2Q33C$

#conn_str_params = {str(pair.split('=')[0]): pair.split('=')[1] for pair in conn_str.split(' ')}
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'momentrosuite-database',
        'HOST': 'momentrosuite-server.postgres.database.azure.com',
        'PORT': '5432',
        'USER': 'wfdfblpion',
        'PASSWORD': 'N78SRBM56II2Q33C$',
    }
}

# STATIC_ROOT = os.path.join(BASE_DIR,'staticfiles')

# connection_string = os.environ['AZURE_POSTGRESQL_CONNECTIONSTRING']
# parameters = {str(pair.split('=')[0]): pair.split('=')[1] for pair in connection_string.split(' ')}

# # DATABASES = {
    
# #     'default':{
# #         'ENGINE': 'django.db.backends.postgresql',
# #         'NAME' :parameters['dbname'],
# #         'HOST':parameters['host'],
# #         'USER':parameters['user'],
# #         'PASSWORD':parameters['password'],
            
# #     }
# # }

# DATABASES={
#    'default':{
#       'ENGINE':'django.db.backends.postgresql_psycopg2',
#       'NAME':os.getenv('DATABASE_NAME'),
#       'USER':os.getenv('DATABASE_USER'),
#       'PASSWORD':os.getenv('DATABASE_PASSWORD'),
#       'HOST':os.getenv('DATABASE_HOST'),
#       'PORT':'5432',
#       'OPTIONS': {'sslmode': 'require'}
#    }
# }

# # DATABASES = {
    
# #     'default':{
# #         'ENGINE': 'django.db.backends.postgresql',
# #         'NAME' :'demo-sandbox',
# #         'HOST':'demo-sandbox.postgres.database.azure.com',
# #         'USER':'madushanjaysri',
# #         'PASSWORD':'@Jayaz1996',
            
# #     }
# # }