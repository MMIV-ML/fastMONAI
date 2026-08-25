FROM haukebartsch/fiona-component-python:latest
#
# Build from deployment/pacs/.
# Release builds use --pull and pass --build-arg VERSION="<UTC build version>".
#

WORKDIR /app

LABEL "com.ror.vendor"="MMIV.no" \
    version="1.0" \
    com.ror.conda.env.name="fastmonai" \
    description="Vestibular schwannoma segmentation for the MMIV Research PACS."

ENV REPORT_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
ENV DCMDICTPATH=/dcmtk/dcmdata/data/dicom.dic:/dcmtk/dcmdata/data/private.dic

# Create the environment:
COPY requirements.yml .
RUN conda env create -f requirements.yml

# VERSION is the required image and runtime stamp used for DICOM provenance.
# Declared after the conda layer so a new version does not rebuild the environment.
ARG VERSION
RUN test -n "${VERSION}" || \
    (echo "VERSION build argument is required" >&2; exit 1)
LABEL org.opencontainers.image.version="${VERSION}"
ENV VERSION="${VERSION}"

COPY deployment_hashing.py deployment_models.py deployment_bundle.py \
    dicom_output.py pacs_inference.py entrypoint.sh ./
COPY model_bundles/ ./model_bundles/

# The code to run when container is started:
RUN ["chmod", "+x", "/app/entrypoint.sh"]
ENTRYPOINT [ "/app/entrypoint.sh" ]
