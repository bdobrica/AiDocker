import os
from collections.abc import Iterable
from typing import Generator, List, Optional, Tuple, Type

import sqlalchemy
from flask import g
from pydantic import BaseModel


def get_db_connection() -> sqlalchemy.Engine:
    """Opens a new database connection if there is none yet for the current application context."""

    db: sqlalchemy.Engine = getattr(g, "_database", None)
    if db is None:
        db_conn_string = os.getenv("DATABASE_CONN_STRING", "sqlite:///database.db")
        db = sqlalchemy.create_engine(db_conn_string, echo=True)
        setattr(g, "_database", db)
    return db


def dispose_db_engine(exception: Exception) -> None:
    """Closes the database again at the end of the request. Intended for use with `flask.Flask.teardown_appcontext`."""
    db: sqlalchemy.Engine = getattr(g, "_database", None)
    if db is not None:
        db.dispose()


class OrmBase(BaseModel):
    """
    Provides a base class for ORM models that can be used to create, select, insert, update and delete records.
    Common properties:
    - id: Optional[int] = None - The primary key of the record. If missing, a new record will be inserted, otherwise the existing record will be updated.
    - __prefix__: str = "" - The prefix to use for the table name. Defaults to the empty string.
    - __table__: sqlalchemy.Table - The table object for the model. This is set automatically when the model is created.
    """

    id: Optional[int] = None
    __prefix__: str = ""

    @classmethod
    def _get_snakecase_name(cls: Type["OrmBase"]) -> str:
        """Converts the class name to snake case. Eg. MyModel -> my_model"""
        return "".join("_" + c.lower() if c.isupper() else c for c in cls.__name__).lstrip("_")

    @classmethod
    def _get_table_name(cls: Type["OrmBase"]) -> str:
        """
        Returns the table name for the model. Can be overridden to customize the table name. Defaults to the snake case
        name of the model prefixed with the value of __prefix__.
        """
        return cls.__prefix__ + cls._get_snakecase_name()

    @classmethod
    @property
    def __db__(cls) -> sqlalchemy.engine:
        """Returns the database connection."""
        return get_db_connection()

    @classmethod
    def _get_columns(cls) -> List[sqlalchemy.Column]:
        """Returns the columns for the table, based on the fields of the model."""
        type_mapping = {
            int: sqlalchemy.Integer,
            str: sqlalchemy.String,
            bool: sqlalchemy.Boolean,
            float: sqlalchemy.Numeric,
        }
        columns = []
        for key, field in cls.__fields__.items():
            if issubclass(field.type_, OrmBase):
                columns.append(
                    sqlalchemy.Column(
                        key,
                        type_mapping.get(field.type_.__fields__.get("id"), sqlalchemy.String),
                        sqlalchemy.ForeignKey(f"{field.type_._get_table_name()}.id"),
                    )
                )
            elif field.type_ in type_mapping:
                columns.append(
                    sqlalchemy.Column(
                        key,
                        type_mapping[field.type_],
                        primary_key=key == "id",
                        autoincrement=key == "id" and field.type_ == int,
                        nullable=field.default is None,
                    )
                )
            else:
                columns.append(
                    sqlalchemy.Column(
                        key,
                        sqlalchemy.String,
                    )
                )
        return columns

    @classmethod
    def create(cls) -> Type["OrmBase"]:
        """Creates the table for the model if it does not exist yet."""
        with cls.__db__.connect() as conn:
            if not cls.__db__.dialect.has_table(conn, cls._get_table_name()):
                metadata = sqlalchemy.MetaData()
                metadata.reflect(bind=conn)
                table = sqlalchemy.Table(cls._get_table_name(), metadata, *cls._get_columns())
                metadata.create_all(cls.__db__)
                setattr(cls.__class__, "__table__", table)
            elif not hasattr(cls.__class__, "__table__"):
                metadata = sqlalchemy.MetaData()
                metadata.reflect(bind=conn)
                table = metadata.tables[cls._get_table_name()]
                setattr(cls, "__table__", table)
        return cls

    @classmethod
    def select(cls, **kwargs) -> Generator["OrmBase", None, None]:
        """Queries the table for records matching the given criteria."""
        if not hasattr(cls, "__table__"):
            cls.create()

        args = []
        for key, value in kwargs.items():
            if key in cls.__fields__:
                if isinstance(value, Iterable) and not isinstance(value, str):
                    args.append(getattr(cls.__table__.c, key).in_(value))
                else:
                    args.append(getattr(cls.__table__.c, key) == value)

        query = sqlalchemy.select(cls.__table__).where(*args)
        with cls.__db__.connect() as conn:
            result = conn.execute(query)
        for row in result.mappings().all():
            yield cls.parse_obj(row)

    @classmethod
    def select_paginated(cls, page: int = 0, per_page: int = 10, **kwargs) -> Tuple[List["OrmBase"], int, int]:
        """Queries the table for records matching the given criteria."""
        if not hasattr(cls, "__table__"):
            cls.create()

        args = []
        for key, value in kwargs.items():
            if key in cls.__fields__:
                if isinstance(value, Iterable) and not isinstance(value, str):
                    args.append(getattr(cls.__table__.c, key).in_(value))
                else:
                    args.append(getattr(cls.__table__.c, key) == value)

        count_query = sqlalchemy.select(sqlalchemy.func.count()).select_from(cls.__table__).where(*args)
        with cls.__db__.connect() as conn:
            result = conn.execute(count_query)
            count = result.scalar()
            pages = count // per_page + 1
            if page < 0:
                page = 0
            elif page > pages:
                page = pages - 1

            query = (
                sqlalchemy.select(cls.__table__)
                .where(*args)
                .order_by(cls.__table__.c.id)
                .offset(page * per_page)
                .limit(per_page)
            )
            result = conn.execute(query)

        return [cls.parse_obj(row) for row in result.mappings().all()], page, pages

    def insert(self) -> "OrmBase":
        """Inserts a new record into the table based on the model."""
        if not hasattr(self.__class__, "__table__"):
            self.create()
        data = {key: getattr(self, key) for key in self.__fields__}
        for key, value in data.items():
            if isinstance(value, OrmBase):
                data[key] = value.id
        query = sqlalchemy.insert(self.__class__.__table__).values(**data)
        with self.__db__.connect() as conn:
            result = conn.execute(query)
            self.id = result.inserted_primary_key[0]
            conn.commit()
        return self

    def update(self) -> "OrmBase":
        """Updates an existing record in the table based on the model."""
        if not hasattr(self.__class__, "__table__"):
            self.create()

        if self.id is None:
            return self.insert()

        query = (
            sqlalchemy.update(self.__table__)
            .where(self.__table__.c.id == self.id)
            .values(**{key: getattr(self, key) for key in self.__fields__ if key != "id"})
        )
        with self.__db__.connect() as conn:
            _ = conn.execute(query)
            conn.commit()
        return self

    def delete(self) -> "OrmBase":
        """Deletes an existing record from the table based on the model."""
        if not hasattr(self.__class__, "__table__") or self.id is None:
            return self

        query = sqlalchemy.delete(self.__table__).where(self.__table__.c.id == self.id)
        with self.__db__.connect() as conn:
            _ = conn.execute(query)
            conn.commit()
        return self
