Expression Properties
---------------------

Blaze expressions adhere to the following properties

1.  They use `__slots__` to ensure a simple model
2.  Their string representations completely recreate the object

        eval(str(expr)) == expr

3.  All information can be found in the `.args` property and their type.  This
drives `__eq__` and `__hash__`.


