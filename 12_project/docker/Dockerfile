FROM public.ecr.aws/lambda/python:3.9

RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY model.tflite .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]