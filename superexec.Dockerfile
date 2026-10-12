FROM flwr/superexec:1.39.0

WORKDIR /sklearnexample

COPY pyproject.toml LICENSE ./
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]