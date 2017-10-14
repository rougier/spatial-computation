"""Python 2/3 compatibility helpers
"""
try:
    from StringIO import StringIO
except:
    from io import StringIO
